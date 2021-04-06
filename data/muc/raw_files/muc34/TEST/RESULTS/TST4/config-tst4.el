;;; FILE NAME  config-tst4.el
;;;
;;; MUC VERSION  	3.3 (note: v3.4 used for MUC-4 rescoring by NRaD)
;;;
;;; TST4 MUC configuration file (assumes tst4 subdirectory)
;;;

 
(define-muc-configuration-options
  :slot-file    (expand-file-name
                  (concat *muc-base-directory* "tst4/slotconfig-tst4.el"))
  :key-file     (expand-file-name
                   (concat *muc-base-directory* "tst4/key-tst4.v2"))
  :response-file (expand-file-name
                  (concat *muc-base-directory* "tst4/your-response-file.tst4"))
  :message-file (expand-file-name
                 (concat *muc-base-directory* "tst4/tst4-muc4"))
  :history-file (expand-file-name
                 (concat *muc-base-directory* "tst4/final-cumulative-hist.tst4"))
  :disable-edit t
  :query-verbose t
  :report-verbose t
  :display-type 'all-templates
  :score-irrelevant t
  :scored-slots :all
  :restrict-map-slots '(and inc-type 
			    (or phys-tgt-id phys-tgt-type 
				hum-tgt-name hum-tgt-desc hum-tgt-type 
				perp-ind-id perp-org-id))
  :scored-templates :all
  :string-premodifiers '("a" "the" "an" 
			 "this" "that" "these" "those"
			 "one" "two" "three" "four" "five" 
			 "six" "seven" "eight" "nine" "ten"
			 "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
			 "more" "most" "many" "several" "some" "all" "few"
			 "any" "another" "other" "certain"
			 "of")
  :subtotal-rows '((inc-total inc-date inc-loc inc-type inc-stage 
			      inc-instr-id inc-instr-type)
		   (perp-total perp-inc-cat perp-ind-id perp-org-id 
			       perp-org-conf)
		   (phys-tgt-total phys-tgt-id phys-tgt-type phys-tgt-num 
				  phys-tgt-nation phys-tgt-effect 
				  phys-tgt-total-num)
		   (hum-tgt-total hum-tgt-name hum-tgt-desc hum-tgt-type
				 hum-tgt-num hum-tgt-nation hum-tgt-effect
				 hum-tgt-total-num)))

