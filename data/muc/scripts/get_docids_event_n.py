import json
import os
from collections import OrderedDict
from collections import Counter
from transformers import BertTokenizer
tag2role = OrderedDict({'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg", 'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})

def num_mentions(entity):
    entity = sorted(entity, key = lambda m : len(m[0]), reverse=True)
    entity_no_overlap = []
    for candidate_m in entity:
        to_add = True
        for m in entity_no_overlap:
            if candidate_m[0] in m[0]:
                to_add = False
        if to_add:
            entity_no_overlap.append(candidate_m)
    return len(entity_no_overlap)
    # return len(entity)

if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # # max num of entitys per article (20)
    # doc_num_of_entitys = []
    # for div in ["train", "dev", "test"]:
    #     data_dir = "../processed/"
    #     file_path = os.path.join(data_dir, "{}.json".format(div))
    #     with open(file_path, encoding="utf-8") as f:
    #         for line in f:
    #             line = json.loads(line)
    #             extracts = line["extracts"]
    #             num_of_entitys = 0
    #             for _, role in tag2role.items():
    #                 num_of_entitys += len(extracts[role])
    #             doc_num_of_entitys.append(num_of_entitys)
    #             if num_of_entitys > 15:
    #                 print(div, "example num_of_entitys > 15")

    # print("max_num_of_entitys (per doc): ", max(doc_num_of_entitys))
    # print("avg_num_of_entitys (per doc): ", sum(doc_num_of_entitys)/len(doc_num_of_entitys))
    # print("distribution: ", Counter(doc_num_of_entitys).most_common) # 0: 794; total: 1700

    # print("========================")
    
    all_entity_num_mention_list = []
    docid_avg_entity_num_mention = OrderedDict()
    docids = {0:[], 1:[], 2:[], 3:[], 4:[], ">=2":[]}
    div = "test"
    data_dir = "../processed/"
    file_path = os.path.join(data_dir, "{}.json".format(div))
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1])
            templates = line["templates"]
            num_events = len(templates)
            if num_events == 0:
                docids[0].append(str(docid))
            elif num_events == 1:
                docids[1].append(str(docid))
            elif num_events == 2:
                docids[2].append(str(docid))
            elif num_events == 3:
                docids[3].append(str(docid))
            elif num_events >= 4:
                docids[4].append(str(docid))

            if num_events >= 2:
                docids[">=2"].append(str(docid))



    avg_entity_num_mention_docid = sorted([(docid, avg_num) for docid, avg_num in docid_avg_entity_num_mention.items()], key = lambda pair: pair[1]) 
    with open("../processed/docids_event_n.json", "w+") as f:
        f.write(json.dumps(docids, indent=4))
    # print("avg # of mentions per entity: {:.2f}".format(sum(all_entity_num_mention_list)/len(all_entity_num_mention_list)))
    # print("median # of mentions per entity: {:.2f}".format(avg_entity_num_mention_docid[len(avg_entity_num_mention_docid)//2][1]))
    for event_n in docids:
        print("num of events ", event_n)
        print(len(docids[event_n]))


    # import ipdb; ipdb.set_trace()

    # for role in role_entity_num_list:
        # print("{} average # entity per role: {:.2f}".format(role, sum(role_entity_num_list[role])/len(role_entity_num_list[role])))