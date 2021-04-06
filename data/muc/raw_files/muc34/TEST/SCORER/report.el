;;;
;;; FILE NAME  report.el
;;;
;;; SCCS ID:         @(#)report.el	6.4 5/12/92
;;; MUC VERSION 3.2
;;;
;;; Copyright (C) 1992 by Science Applications International Corporation
;;;
;;; This file is part of a software package that has over 40% new source lines
;;; of code by SAIC.
;;;
;;; The original MUC source code was provided by Naval Ocean Systems Center to
;;; SAIC under contract N66001-90-D-0192, and was originally authored by
;;; Pete Halverson of General Electric.
;;;
;;; Modifications to this file have been directed by Nancy Chinchor
;;; (Project Manager) and implemented by Shannon Torrey (Software Engineer)
;;; of SAIC.
;;;
;;; Please send comments or discrepancy reports to chinchor@esosun.css.gov
;;; and shannon@esosun.css.gov.
;;;
;;; SYNOPSIS
;;;       
;;;       
;;;
;;; DESCRIPTION
;;;	
;;;       
;;;       
;;;
;;;       
;;;
;;; DIAGNOSTICS
;;;       Describe any problems, hidden pitfalls or subtle assumptions.
;;;       Specify error conditions and results.
;;;
;;; FILES
;;;  	
;;;       
;;;       
;;;
;;; NOTES
;;;      	This is a scratch-pad area.  Use for notes re future enhancements,
;;;     	algorithm descriptions, history, etc.
;;;
;;; SEE ALSO
;;;
;;;
;;; AUTHOR/DATE  (st)  13:48:33 02/20/92
;;;
;;;
;;; MODIFICATIONS
;;;	(ty)  10:11:38 05/01/92 Replaced average computation with F Measures
;;;	(ty)  08:08:24 05/12/92 Moved F-Measures text down a line in score.
;;;
;;;
;;;
;;;

(defvar *suppress-abnormal-scores* nil
  "If non-null, perfect recall, precision, fallout, and overgeneration
scores are not included in individual template slot reports.")

(defvar *display-type* nil
  "If non-null, use to determine which score totals to show on detailed
reports - missing, spurious, noncommital, or all templates ")

(defun display-single-template-tallies (key response status)
  "Display the results of scoring a single key/response template pair in
a pop-up window buffer."
  (prepare-score-display)
  (format-single-template-tallies key response status)
  (show-score-display))

(defun prepare-score-display (&optional retain-p)
  "Creates the score result buffer, in preparation for a
scoring run.  Call show-score-display when you're ready to actually
show the buffer."
  (setq *score-output-buffer* (get-buffer-create "*MUC Score Display*"))
  (set-buffer *score-output-buffer*)
  (setq buffer-read-only nil)
  (prepare-score-display-aux retain-p))

(defun prepare-score-display-aux (&optional retain-p)
  "Initializes the score result buffer, in preparation for a
scoring run."
  (unless retain-p
    (erase-buffer)
    (insert (format "Configuration: %s\n\n" *config-filename*))
    (when *score-irrelevant-templates*
      (insert "Incorrect irrelevant response templates are scored.\n\n"))
    (when *disable-buffer-editing*
      (insert "Buffer editing is disabled.\n\n"))
    (if *display-type*
	(insert (format "Report of %s scores\n\n" *display-type*))
      (insert (format "Report of %s scores\n\n" 'matched-missing)))))      

(defun show-score-display ()
  "Show the (presumably filled) score result buffer."
  (muc-browse-mode-1)
  (pop-up-muc-io-window *score-output-buffer* *message-buffer*))
  
(defvar *score-report-slot-column-width* 20
  "The width of the slot-id column in an extended score report.")

(defvar *score-report-final-slot-column* 50
  "The number of characters from the end of the slot-id column to the beginning of the recall column")

(defun format-template-ids (msg-id key response status)
  (insert msg-id)
  (indent-to *score-report-slot-column-width*)
  (insert  " key: "       (if key      (template-id key)      "<NONE>")
           "  response: " (if response (template-id response) "<NONE>"))
  (indent-to (- 79 (length status)))
  (insert status "\n"))

(defun format-message-ids (msg-id status)
  (insert msg-id)
  (when (string= status "[irrelevant]")
      (indent-to (- 79 (length status)))
      (insert status))
  (insert "\n"))

(defun format-slot-groups (slot-list &optional suppress-normal)
  "Given a list of slot-tally structures in SLOT-LIST, display each
one on a single line."
  (while slot-list
    (format-slot-score-tallies (symbol-name (slot-tally-name (car slot-list)))
			       (slot-tally-values (car slot-list))
			       (not (memq (slot-tally-type (car slot-list))
					  '(element boolean)))
			       suppress-normal)
    (setq slot-list (cdr slot-list))))

(defun format-single-template-tallies (key response status)
  "Insert the results of scoring a single key/response template pair
into the current buffer.  Either key or response may be nil, in which
case only the TEMPLATE-ID slot is reported."
  (format-template-ids (if key (template-message-id key)
                               (template-message-id response))
                       key response status)
  (insert "\n")
  (format-slot-score-headings)
  (cond
   ((null response)
    (format-slot-score-tallies ;; 1 possible, 1 miss
       (symbol-name 'template-id) (list 1 0 0 0 0 0 0 0 1 0 0) nil))
   ((null key)
    (format-slot-score-tallies ;; 1 actual, 1 spurious
       (symbol-name 'template-id) (list 0 1 0 0 0 0 0 1 0 0 0) nil))
   (t
    (let ((slots (copy-sequence (slot-scores-slots *current-score*))))
      (format-slot-groups (delq (find-slot-tally 'message-id slots) slots)
			  *suppress-abnormal-scores*))
    (when (slot-scores-subtotals *current-score*)
      (format-slot-score-hline)
      (format-slot-groups (slot-scores-subtotals *current-score*)))
    (format-slot-score-hline)
    (format-slot-score-tallies 
     "TOTAL" (slot-tally-values (slot-scores-totals *current-score*)) t))))


(defun format-cumulative-slot-tallies (&optional show-text-filtering-p)
  "Insert the results of a full scoring session into the current buffer.
Show the TEXT FILTERING row only if SHOW-TEXT-FILTERING-P is non-nil"
  (insert (format " * * * TOTAL SLOT SCORES * * *\n\n"))
  (format-slot-score-headings)
  ;; show the template-id score from "all-templates" 
  (format-slot-score-tallies
   "template-id" 
   (slot-tally-values 
    (find-slot-tally 
     'template-id 
     (slot-scores-slots
      (slots-all-scores-all-templates *cumulative-totals*)))) t)
  (let* ((instance (cond ((eq *display-type* 'matched-spurious)
			  (slots-all-scores-matched-spurious
			   *cumulative-totals*))
			 ((eq *display-type* 'matched-only)
			  (slots-all-scores-matched-only *cumulative-totals*))
			 ((eq *display-type* 'all-templates)
			  (slots-all-scores-all-templates *cumulative-totals*))
			 (t ; default is matched missing
			  (slots-all-scores-matched-missing 
			   *cumulative-totals*))))
	 (slots (copy-sequence (slot-scores-slots instance)))
	 (subtotals (copy-sequence (slot-scores-subtotals instance))))
    (setq slots (delq (find-slot-tally 'message-id slots) slots))
    (setq slots (delq (find-slot-tally 'template-id slots) slots))
    (format-slot-groups slots)
    (when subtotals 
      ;; display the subtotals
      (format-slot-score-hline)
      (format-slot-groups subtotals))
    (format-slot-score-hline)
    (format-slot-score-tallies
     "MATCHED/MISSING" 
     (slot-tally-values 
      (slot-scores-totals
       (slots-all-scores-matched-missing *cumulative-totals*))) t)
    (format-slot-score-tallies
     "MATCHED/SPURIOUS" 
     (slot-tally-values 
      (slot-scores-totals 
       (slots-all-scores-matched-spurious *cumulative-totals*))) t)
    (format-slot-score-tallies
     "MATCHED ONLY" 
     (slot-tally-values 
      (slot-scores-totals 
       (slots-all-scores-matched-only *cumulative-totals*))) t)
    (format-slot-score-tallies
     "ALL TEMPLATES" 
     (slot-tally-values 
      (slot-scores-totals 
       (slots-all-scores-all-templates *cumulative-totals*))) t)
    (format-slot-score-tallies
     "SET FILLS ONLY" 
     (slot-tally-values 
      (slot-scores-set-fill-totals 
       (slots-all-scores-matched-missing *cumulative-totals*))) nil)
    (format-slot-score-tallies
     "STRING FILLS ONLY" 
     (slot-tally-values 
      (slot-scores-string-fill-totals 
       (slots-all-scores-matched-missing *cumulative-totals*))) t)
    (when show-text-filtering-p
      (format-slot-score-tallies 
       "TEXT FILTERING" (slots-all-scores-contingency-tallies 
			 *cumulative-totals*)
       nil nil t)))
;  (format-final-score *cumulative-totals*)
  (format-f-measures
   (slot-tally-values
    (slot-scores-totals 
     (slots-all-scores-all-templates *cumulative-totals*)))))

(defun format-message-template-tallies (message-id status)
  "Insert the results of scoring a message with all of its key/response
template pairs into the current buffer."
  (format-message-ids message-id status)
  (insert "\n")
  (format-slot-score-headings)
  ;; show the template-id score from "all-templates" 
  (format-slot-score-tallies
   "template-id" 
   (slot-tally-values 
    (find-slot-tally 
     'template-id 
     (slot-scores-slots
      (slots-all-scores-all-templates *message-level-totals*)))) t)
  ;; display the individual slots
  (let* ((instance (cond ((eq *display-type* 'matched-spurious)
			  (slots-all-scores-matched-spurious
			   *message-level-totals*))
			 ((eq *display-type* 'matched-only)
			  (slots-all-scores-matched-only 
			   *message-level-totals*))
			 ((eq *display-type* 'all-templates)
			  (slots-all-scores-all-templates 
			   *message-level-totals*))
			 (t ; default is matched missing
			  (slots-all-scores-matched-missing 
			   *message-level-totals*))))
	 (slots (copy-sequence (slot-scores-slots instance)))
	 (subtotals (copy-sequence (slot-scores-subtotals instance))))
    (setq slots (delq (find-slot-tally 'message-id slots) slots))
    (setq slots (delq (find-slot-tally 'template-id slots) slots))
    (format-slot-groups slots)
    ;; display the subtotals
    (when (slot-scores-subtotals instance)
      (format-slot-score-hline)
      (format-slot-groups (slot-scores-subtotals instance)))
    (format-slot-score-hline)
    ;; display the totals
    (format-slot-score-tallies
     "MATCHED/MISSING" 
     (slot-tally-values 
      (slot-scores-totals
       (slots-all-scores-matched-missing *message-level-totals*))) t)
    (format-slot-score-tallies
     "MATCHED/SPURIOUS" 
     (slot-tally-values 
      (slot-scores-totals 
       (slots-all-scores-matched-spurious *message-level-totals*))) t)
    (format-slot-score-tallies
     "MATCHED ONLY" 
     (slot-tally-values 
      (slot-scores-totals 
       (slots-all-scores-matched-only *message-level-totals*))) t)
    (format-slot-score-tallies
     "ALL TEMPLATES" 
     (slot-tally-values 
      (slot-scores-totals 
       (slots-all-scores-all-templates *message-level-totals*))) t)) 
;  (format-final-score *message-level-totals*)
  (format-f-measures
   (slot-tally-values
    (slot-scores-totals 
     (slots-all-scores-all-templates *message-level-totals*)))))

(defun format-slot-score-headings ()
  "Insert the column headings for a slot scoring report."
  (insert "SLOT")
  (indent-to *score-report-slot-column-width*)
  (insert "  POS ACT|COR PAR INC|ICR IPA|SPU MIS NON|REC PRE OVG FAL\n")
  (format-slot-score-hline))

(defun format-slot-score-hline ()
  (insert "-----------------------------+-----------+-------+-----------+---------------\n"))

(defun format-computed-measure (value normal-value suppress-normal)
  (if (and suppress-normal (or (null value) (= value normal-value)))
      (insert "    ")
    (if value
        (insert (format "%3d " value))
      (insert "  * "))))


(defun format-slot-score-tallies (slot-id tallies
                                    &optional suppress-fallout suppress-normal
				    suppress-percentages)
  "Insert a set of score TALLIES for slot SLOT-ID. Compute recall,
precision, and overgeneration scores if SUPPRESS-PERCENTAGES is not given.  
If SUPPRESS-FALLOUT is
nil, also computes fallout score.  If SUPPRESS-NORMAL is non-null,
only abnormal derived scores are reported." 
  (insert slot-id)
  (indent-to *score-report-slot-column-width*)
  (if (null tallies)
      (insert " <no score>\n")
    (let ((recall (apply 'compute-recall tallies))
          (precision (apply 'compute-precision tallies))
          (overgen (apply 'compute-overgen tallies))
          (fallout (and (not suppress-fallout)
                        (apply 'compute-fallout tallies)))
          (column 2))
      (insert (format " %4d " (car tallies)))
      (setq tallies (cdr tallies))
      (if (string-equal slot-id "TEXT FILTERING")
	  (while (cdr tallies);; don't display POSSIBLE-INCORRECT tally
	    (if (and (>= column 4) (<= column 7))
		(insert "  *")
	      (insert (format "%3d" (car tallies))))
	    (if (memq column '(2 5 7 10))
		(insert "|")
	      (insert " "))
	    (setq tallies (cdr tallies) column (1+ column)))
	(while (cdr tallies);; don't display POSSIBLE-INCORRECT tally
	  (insert (format "%3d" (car tallies)))
	  (if (memq column '(2 5 7 10))
	      (insert "|")
	    (insert " "))
	  (setq tallies (cdr tallies) column (1+ column))))
      (unless suppress-percentages
	(format-computed-measure recall 100 suppress-normal)
	(format-computed-measure precision 100 suppress-normal)
	(format-computed-measure overgen 0 suppress-normal)
	(unless suppress-fallout
	  (format-computed-measure fallout 0 suppress-normal)))
      (insert "\n"))))
            
(defun format-f-measures (tallies)
  "Insert 3 f-measures into the final score"
  (let ((recall (apply 'compute-recall tallies))
	(precision (apply 'compute-precision tallies)))
    (if (or (null recall) (null precision))
;;; Changed "F MEASURES" to "F-MEASURES" (ty)  17:16:02 05/19/92
	(insert "F-MEASURES not available, Recall or Precision Nil\n")
      (let* ((recall (f recall))
	     (precision (f precision))
	     (_f4/5 (f/ (f 4) (f 5)))
	     (_f1/5 (f/ _f1 (f 5)))
	     (f1 (compute-f-value _f4/5 precision recall)) 
	     (f2 (compute-f-value _f1/2 precision recall))
	     (f3 (compute-f-value _f1/5 precision recall)))
	(format-slot-score-hline)
	(indent-to *score-report-final-slot-column*)
	(insert "   P&R      2P&R      P&2R\n")
	(insert "F-MEASURES")
	(indent-to (- *score-report-final-slot-column* 2))
	(insert (format "%9s " (remove-trailing-zeros
				(float-to-string (round-to-2-places f2)))))
	(insert (format "%9s " (remove-trailing-zeros
				(float-to-string (round-to-2-places f1)))))
	(insert (format "%9s\n" (remove-trailing-zeros
				 (float-to-string (round-to-2-places f3)))))))))


;;; Fixed divide by 0 bug found by PRC. (ty)  17:16:44 05/19/92
;;;(defun compute-f-value (a p r)
;;;  "Compute 1/(a*(1/p) + (1-a)*(1/r)) where all values are funky float format"
;;;  (f/ _f1 (f+ (f* a (f/ _f1 p)) (f* (f- _f1 a) (f/ _f1 r)))))

(defun compute-f-value (a p r)
  "Compute 1/(a*(1/p) + (1-a)*(1/r)) where all values are funky float format"
  (if (or (fzerop p) (fzerop r))
      (f 0)
    (f/ _f1 (f+ (f* a (f/ _f1 p)) (f* (f- _f1 a) (f/ _f1 r))))))

(defun round-to-2-places (fnum)
  (let* ((_f100 (f 100))
	 (fnum-100 (f* fnum _f100))
	 (ft (ftrunc fnum-100))
	(ftmp nil))
    (setq ftmp (f- fnum-100 ft))
    (if (or (f> ftmp _f1/2)
	    (and (f= ftmp _f1/2) (oddp (fint ft))))
	  (setq ft (f+ ft _f1)))
    (f/ ft _f100)))

;;; Fixed handling of "0" case; related to divide by 0 bug. (ty)  17:17:25 05/19/92
;;;(defun remove-trailing-zeros (string)
;;;  (let ((len (length string)))
;;;    (while (= (aref string (- len 1)) 48)
;;;      (setq string (substring string 0 (- len 1)))
;;;      (setq len (- len 1))))
;;;  string)

(defun remove-trailing-zeros (string)
  (let ((len (length string))
	(period-flag nil))
    (do ((i 1 (+ i 1)))
	((or period-flag (= i len)))
      (if (= (aref string i) 46)   ; period
	  (setq period-flag t)))
    (if period-flag
	(let ((len (length string)))
	  (while (and (= (aref string (- len 1)) 48)	; 0
		      (/= (aref string (- len 2)) 46))  ; period
	    (setq string (substring string 0 (- len 1)))
	    (setq len (- len 1))))))
  string)

;;;
;;; This is now obselete, but keep it for a while.  (ty)  16:06:10 05/05/92
;;;
(defun format-final-score (instance)
  "Insert the final-score line into the scoring buffer"
  (format-slot-score-hline)
  (insert "AVERAGE SCORE")
  (indent-to *score-report-slot-column-width*)
  (insert "                                          ")
  (insert  (format "%3d " 
	     (slots-all-scores-ave-recall-score instance)))
  (insert  (format "%3d\n" 
	     (slots-all-scores-ave-precision-score instance)))
  )

