import argparse
import glob
import logging
import os
import json
from collections import OrderedDict

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from transformer_base import BaseTransformer, add_generic_args, generic_train
from utils_gtt import convert_examples_to_features, get_labels, read_examples_from_file, read_golds_from_test_file, not_sub_string, incident_token_to_type

role_list = ["incident_type", "PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class NERTransformer(BaseTransformer):
    """
    A training module for single-transformer-ee. See BaseTransformer for the core options.
    """

    mode = "base"

    def __init__(self, hparams):
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        # super(NERTransformer, self).__init__(hparams, num_labels, self.mode)
        super(NERTransformer, self).__init__(hparams, self.mode)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.hparams.n_gpu else "cpu")
        # n_gpu = torch.cuda.device_count()
        # self.MASK = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.SEP = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.CLS = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]   
        self.SEP_template = self.tokenizer.convert_tokens_to_ids(["[unused0]"])[0]

    def forward(self, **inputs):
        labels = inputs.pop("labels", None) # doc_length
        args = self.hparams

        outputs = self.model(**inputs) # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        src_sequence_output = sequence_output[:, :args.max_seq_length_src, :]
        src_sequence_output = torch.transpose(src_sequence_output, 1, 2) # hidden * doc_length
        tgt_sequence_output = sequence_output[:, args.max_seq_length_src:, :]  # tgt_length * hidden
        logits = torch.bmm(tgt_sequence_output, src_sequence_output) # tgt_length * doc_length

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            tgt_attention_mask_1d = inputs["attention_mask"][:, -1, args.max_seq_length_src:]
            if tgt_attention_mask_1d is not None:
                active_logits = logits.view(-1, args.max_seq_length_src)
                active_labels = labels.view(-1)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, args.max_seq_length_src), labels.view(-1))
            outputs = (loss,) + outputs

        # import ipdb; ipdb.set_trace()
        return outputs

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "position_ids": batch[3], "labels": batch[4]}
        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {"training_loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file):
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = read_examples_from_file(args.data_dir, mode, self.tokenizer, debug=args.debug)
                features = convert_examples_to_features(
                    examples,
                    # self.labels,
                    args.max_seq_length_src,
                    args.max_seq_length_tgt,
                    self.tokenizer,
                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=bool(args.model_type in ["roberta"]),
                    pad_on_left=bool(args.model_type in ["xlnet"]),
                    pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
        # import ipdb; ipdb.set_trace()

    def load_dataset(self, mode, batch_size):
        "Load datasets. Called after prepare data."
        args = self.hparams
        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if args.debug:
            features = features[:2]
            # features = features[:len(features)//10]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_docid = torch.tensor([f.docid for f in features], dtype=torch.long)
        return DataLoader(
            TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_position_ids, all_label_ids, all_docid), batch_size=batch_size
        )

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "position_ids": batch[3], "labels": batch[4]}
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        docid = batch[5].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids, "docid": docid}


    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(out_label_ids[i][j])
                    preds_list[i].append(preds[i][j])
        # import ipdb; ipdb.set_trace()

        logs = {
            "val_loss": val_loss_mean,
            "val_accuracy": accuracy_score(out_label_list, preds_list)
        }
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}


    def test_step(self, batch, batch_nb):
        "Compute test"
        # test_loss
        # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "position_ids": batch[3], "labels": batch[4]}
        # outputs = self(**inputs)
        # tmp_eval_loss, tmp_eval_logits = outputs[:2]
        # tmp_eval_logits = tmp_eval_logits.detach().cpu().numpy()
        # out_label_ids = inputs["labels"].detach().cpu().numpy()

        # preds (real decoding)
        max_seq_length_src = self.hparams.max_seq_length_src
        max_seq_length_tgt = self.hparams.max_seq_length_tgt
        bs = batch[0].size(0)

        i = max_seq_length_src
        src_input_ids = batch[0][:, :max_seq_length_src]
        src_position_ids = batch[3][:, :max_seq_length_src]
        tgt_input_ids, init_tgt_input_ids = torch.tensor([[self.CLS]]).to(self.device), torch.tensor([[self.CLS]]).to(self.device)
        tgt_position_ids, init_tgt_position_ids = torch.tensor([[0]]).to(self.device), torch.tensor([[0]]).to(self.device)

        # get out_input_id_list (pred_seq)
        while i <= max_seq_length_src + max_seq_length_tgt - 1:
            input_ids = torch.cat((src_input_ids, tgt_input_ids), dim=1)
            attention_mask = batch[1][:, :i+1, :i+1]
            for j in range(max_seq_length_src, i+1):
                attention_mask[:, j, max_seq_length_src:j+1] = 1
            # if i == max_seq_length_src + 3: # debug
                # import ipdb; ipdb.set_trace()
            token_type_ids = batch[2][:, :i+1]
            position_ids = torch.cat((src_position_ids, tgt_position_ids), dim=1)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "position_ids": position_ids}
            # print(tgt_position_ids) # debug
            outputs = self(**inputs)
            logits = outputs[0][0]

            ## option 1: setting the decoding constraints (!!!)
            # # (constraint 1) on decoding offset (length and larger offset)
            # for j in range(tgt_position_ids.size(1)):
            #     if j == 0: continue
            #     cur_token_id = tgt_input_ids[0][j].detach().cpu().tolist()
            #     cur_token_position = tgt_position_ids[0][j].detach().cpu().tolist()              
            #     if cur_token_id == self.CLS or cur_token_id == self.SEP or cur_token_id == self.SEP_template: 
            #         continue
            #     else:
            #         # remove (early stop/output [SEP]) the case like ``the post post post post ...''
            #         token_id_cnt = 0
            #         k = j
            #         while k > 0 and tgt_input_ids[0][k].detach().cpu().tolist() == cur_token_id:
            #             token_id_cnt += 1
            #             k -= 1
            #         if token_id_cnt >= 4:
            #             for q in range(max_seq_length_src):
            #                 if src_input_ids[0][q].detach().cpu().tolist() == self.SEP:
            #                     logits[j][q] += 10000.0

            #     before_mask = cur_token_position
            #     for k in range(before_mask):
            #         if src_input_ids[0][k].detach().cpu().tolist() == self.SEP: 
            #             continue
            #         logits[j][k] -= 10000.0
            #     # for k in range(cur_token_position + 30, max_seq_length_src):
            #         # if src_input_ids[0][k].detach().cpu().tolist() == self.SEP: continue
            #         # logits[j][k] -= 10000.0

            # (constraint 2) thresh for predicting [SEP]
            probs = torch.nn.Softmax(dim=-1)(logits)
            top_2_probs, top_2_indices = torch.topk(probs, 2, dim=-1)
            for j in range(top_2_indices.size(0)):
                prob_gap = (top_2_probs[j][0]/top_2_probs[j][1]).detach().cpu().tolist()
                if src_input_ids[0][top_2_indices[j][0].detach().cpu().tolist()].detach().cpu().tolist() == self.SEP and prob_gap < global_args.thresh:
                    top_2_indices[j][0] = top_2_indices[j][1]

            out_position_id = top_2_indices[:, 0]

            # # option 2: direct greedy decoding
            # out_position_id = torch.argmax(logits, -1)
            
            # print(out_position_id) # debug
            out_input_id = torch.index_select(src_input_ids, 1, out_position_id)
            out_position_id = out_position_id.unsqueeze(dim=0) # add batch dim
            tgt_input_ids = torch.cat((init_tgt_input_ids, out_input_id), dim=1)
            tgt_position_ids = torch.cat((init_tgt_position_ids, out_position_id), dim=1)
            i += 1

        # from out_input_id_list (pred_seq) to pred_extracts
        docids = batch[5].detach().cpu().tolist()
        pred_seq = []
        pred_extract = []
        for b in range(bs): # bs == 1
            src_input_id_list = src_input_ids[b].detach().cpu().tolist()
            out_input_id_list = out_input_id[b].detach().cpu().tolist()
            out_position_id_list = out_position_id[b].detach().cpu().tolist()
            if out_input_id_list[-1] != self.CLS:
                out_input_id_list.append(self.CLS)

            # get raw pred_seq
            # sep_cnt = 0
            for idx, token_id in enumerate(out_input_id_list):
                if token_id == self.CLS:
                    break
            pred_seq.append(self.tokenizer.convert_ids_to_tokens(out_input_id_list[:idx+1]))

            # get pred_extract
            temps_extract = []
            buf_template = []
            buf_template_pos = []
            for idx, token_id in enumerate(out_input_id_list[:idx]):
                if token_id == self.SEP_template:
                    # decode one template's content
                    # incident_token_to_type[]
                    p_extract = []
                    sep_cnt = 0
                    position_buf = []
                    for temp_idx, temp_token_id in enumerate(buf_template):
                        if temp_token_id == self.SEP:
                            sep_cnt += 1
                            entitys = []
                            s_e_pair = []
                            for position in position_buf:
                                s_e_pair.append(position)
                                if len(s_e_pair) == 2:
                                    s, e = s_e_pair[0], s_e_pair[1]
                                    extract_ids = []
                                    for j in range(s, e+1): 
                                        extract_ids.append(src_input_id_list[j])
                                    extract_tokens = self.tokenizer.convert_ids_to_tokens(extract_ids)
                                    if extract_tokens:
                                        if len(extract_tokens) <= 20: 
                                            candidate_str = " ".join(extract_tokens).replace(" ##", "").replace(" - ", "-")
                                            if sep_cnt != 5 or "bomb" not in candidate_str:
                                                if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
                                                    entitys.append([candidate_str])
                                    s_e_pair = []
                            # extra s in s_e_pair
                            if s_e_pair:
                                extract_tokens = self.tokenizer.convert_ids_to_tokens([src_input_id_list[s_e_pair[0]]])
                                if len(extract_tokens) <= 20: 
                                    candidate_str = " ".join(extract_tokens).replace(" ##", "").replace(" - ", "-")
                                    if sep_cnt != 5 or "bomb" not in candidate_str:
                                        if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
                                            entitys.append([candidate_str])
                            # add all entitys of this role
                            p_extract.append(entitys)
                            # clean buffer
                            position_buf = []
                        else:
                            position_buf.append(buf_template_pos[temp_idx])
                        if sep_cnt >= 6: break

                    # extra token1 token2 [unused0] (no final [SEP])
                    if position_buf:
                        entitys = []
                        s_e_pair = []
                        for position in position_buf:
                            s_e_pair.append(position)
                            if len(s_e_pair) == 2:
                                s, e = s_e_pair[0], s_e_pair[1]
                                extract_ids = []
                                for j in range(s, e+1): 
                                    extract_ids.append(src_input_id_list[j])
                                extract_tokens = self.tokenizer.convert_ids_to_tokens(extract_ids)
                                if extract_tokens:
                                    if len(extract_tokens) <= 20: 
                                        candidate_str = " ".join(extract_tokens).replace(" ##", "").replace(" - ", "-")
                                        if sep_cnt != 5 or "bomb" not in candidate_str:
                                            if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
                                                entitys.append([candidate_str])
                        p_extract.append(entitys)

                    if p_extract:
                        temps_extract.append(p_extract)
                    buf_template = []
                    buf_template_pos = []
                else:
                    buf_template.append(out_input_id_list[idx])
                    buf_template_pos.append(out_position_id_list[idx])


            pred_extract.append(temps_extract)
            ### old ###
            # sep_cnt = 0
            # position_buf = []
            # for idx, token_id in enumerate(out_input_id_list):
            #     if token_id == self.SEP:
            #         sep_cnt += 1
            #         entitys = []
            #         s_e_pair = []
            #         for position in position_buf:
            #             s_e_pair.append(position)
            #             if len(s_e_pair) == 2:
            #                 s, e = s_e_pair[0], s_e_pair[1]
            #                 extract_ids = []
            #                 for j in range(s, e+1): 
            #                     extract_ids.append(src_input_id_list[j])
            #                 extract_tokens = self.tokenizer.convert_ids_to_tokens(extract_ids)
            #                 if extract_tokens:
            #                     if len(extract_tokens) <= 20: 
            #                         candidate_str = " ".join(extract_tokens).replace(" ##", "")
            #                         if sep_cnt != 4 or "bomb" not in candidate_str:
            #                             if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
            #                                 entitys.append([candidate_str])
            #                 s_e_pair = []
            #         # extra s in s_e_pair
            #         if s_e_pair:
            #             extract_tokens = self.tokenizer.convert_ids_to_tokens([src_input_id_list[s_e_pair[0]]])
            #             if len(extract_tokens) <= 20: 
            #                 candidate_str = " ".join(extract_tokens).replace(" ##", "")
            #                 if sep_cnt != 4 or "bomb" not in candidate_str:
            #                     if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
            #                         entitys.append([candidate_str])
            #         # add all entitys of this role
            #         p_extract.append(entitys)
            #         # clean buffer
            #         position_buf = []
            #     else:
            #         position_buf.append(out_position_id_list[idx])
            #     if sep_cnt >= 5: break
            ### old ###


        # return {"test_loss": tmp_eval_loss.detach().cpu(), "pred_seq": pred_seq, "pred_extract": pred_extract, "logits": tmp_eval_logits, "target": out_label_ids, "docid": docids}
        return {"pred_seq": pred_seq, "pred_extract": pred_extract, "docid": docids}


    def test_epoch_end(self, outputs):
        # # updating to test_epoch_end instead of deprecated test_end
        args = self.hparams
        logs = {}

        ## real decoding
        # read golds
        doctexts_tokens, golds = read_golds_from_test_file(args.data_dir, self.tokenizer, debug=args.debug)
        # get preds and preds_log
        preds = OrderedDict()
        preds_log = OrderedDict()
        for x in outputs:
            docids = x["docid"]
            pred_seq = x["pred_seq"]
            pred_extract = x["pred_extract"]
            # preds (pred_extract)]
            for docid, temps_extract in zip(docids, pred_extract):
                if docid not in preds:
                    preds[docid] = []
                    for temp_raw in temps_extract:
                        temp = OrderedDict()
                        for idx, role in enumerate(role_list):
                            temp[role] = []
                            if idx+1 > len(temp_raw):
                                continue
                            elif temp_raw[idx]:
                                if role == "incident_type":
                                    if temp_raw[idx][0][0] in incident_token_to_type:
                                        temp[role] = incident_token_to_type[temp_raw[idx][0][0]]
                                    else:
                                        temp[role] = temp_raw[idx][0][0]
                                else:
                                    temp[role] = temp_raw[idx]

                        preds[docid].append(temp)

                            
            # preds_log
            for docid, p_seq in zip(docids, pred_seq):
                if docid not in preds_log:
                    preds_log[docid] = OrderedDict()
                    preds_log[docid]["doctext"] = " ".join(doctexts_tokens[docid])
                    preds_log[docid]["pred_seq"] = " ".join(p_seq)
                    preds_log[docid]["pred_templates"] = preds[docid]
                    preds_log[docid]["gold_templates"] = golds[docid]

        # # evaluate
        # results = eval_ceaf(preds, golds)
        # logger.info("================= CEAF score =================")
        # logger.info("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
        # logger.info("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
        # logger.info("==============================================")
        # logs["test_micro_avg_f1_phi_strict"] = results["strict"]["micro_avg"]["f1"]

        logger.info("writing preds to .out file:")
        if args.debug:
            with open("preds_gtt_debug.out", "w+") as f:
                f.write(json.dumps(preds_log, indent=4))            
        else:
            with open("preds_gtt.out", "w+") as f:
                f.write(json.dumps(preds_log, indent=4))

        # import ipdb; ipdb.set_trace()

        return {"log": logs, "progress_bar": logs}
        # return {"test_loss": logs["test_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length_src",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization for src. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--max_seq_length_tgt",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization for tgt. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        parser.add_argument("--debug", action="store_true", help="if in debug mode")

        parser.add_argument("--thresh", default=1, type=float, help="thresh for predicting [SEP]",)
        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = NERTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    global_args = args
    logger.info(args)
    model = NERTransformer(args)
    trainer = generic_train(model, args)

    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = NERTransformer.load_from_checkpoint(checkpoints[-1])
        if args.debug:
            model.hparams.debug = True
        trainer.test(model)
