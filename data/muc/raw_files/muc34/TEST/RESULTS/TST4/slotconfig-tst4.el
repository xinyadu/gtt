;;;
;;; FILE NAME  slotconfig.el
;;;
;;; MUC VERSION  	3.3 (note: v3.4 used for MUC-4 rescoring by NRaD)
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
;;; Modifications to this file have been made by Nancy Chinchor
;;; of SAIC.
;;;
;;; Please send comments or discrepancy reports to chinchor@esosun.css.gov
;;;
;;; SYNOPSIS
;;;       
;;;       
;;;
;;; DESCRIPTION
;;;	
;;; MUC Message Template Slots
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
;;;      	This is a scratch-pad area.  Use for notes re future 
;;;		enhancements, algorithm descriptions, history, etc.
;;;
;;; SEE ALSO
;;;
;;;
;;; AUTHOR/DATE  (st)  15:42:13 01/03/92
;;;
;;;
;;; MODIFICATIONS
;;;		 (nc)  12:48:48 03/08/92
;;;			Changed file to conform to MUC 4
;;;              (nc)  08:50:43 05/19/92
;;;                     Added to foreign nations
;;;              (nc)  17:39:03 05/24/92
;;;                     Updated for TST4-MUC4

(setq *message-header-regexp* "^TST4-MUC4-")
(setq *template-header-regexp* "^0.  MESSAGE: ID")
(setq *template-tail-regexp* "^[ \t]*$")

(define-muc-template-slots
  (message-id         "0.  MESSAGE: ID" id)
  (template-id        "1.  MESSAGE: TEMPLATE" template-id)
  (inc-date           "2.  INCIDENT: DATE" date)
  (inc-loc            "3.  INCIDENT: LOCATION" location)
  (inc-type           "4.  INCIDENT: TYPE" element
                      (set ("ATTACK"
			     "ARSON"
			     "BOMBING"
			     "KIDNAPPING"
			     "HIJACKING"
			     "ROBBERY"
			     "FORCED WORK STOPPAGE")))
  (inc-stage          "5.  INCIDENT: STAGE OF EXECUTION" element
                      (set "ACCOMPLISHED" 
			   "ATTEMPTED"
			   "THREATENED"))
  (inc-instr-id       "6.  INCIDENT: INSTRUMENT ID" string)
  (inc-instr-type     "7.  INCIDENT: INSTRUMENT TYPE" element
                      (set ("GUN"
                             "MACHINE GUN"
                             "MORTAR"
                             "HANDGUN"
                             "RIFLE")
                           ("EXPLOSIVE"
                             ("BOMB"
                                "VEHICLE BOMB"
                                "DYNAMITE"
                                "MINE"
                                "AERIAL BOMB")
                             "GRENADE"
                             "MOLOTOV COCKTAIL")
			   ("PROJECTILE"
			     "MISSILE"
			     "ROCKET")
                           "CUTTING DEVICE"
                           "FIRE"
			   "STONE"
                           "TORTURE")
		      (xref inc-instr-id))
  (perp-inc-cat       "8.  PERP: INCIDENT CATEGORY" element
                      (set "TERRORIST ACT"
			   "STATE-SPONSORED VIOLENCE"))
  (perp-ind-id        "9.  PERP: INDIVIDUAL ID" string)
  (perp-org-id        "10. PERP: ORGANIZATION ID" string
		      (xref perp-ind-id))
  (perp-org-conf      "11. PERP: ORGANIZATION CONFIDENCE" element
		      (set "REPORTED AS FACT"
			   "ACQUITTED"
			   "CLAIMED OR ADMITTED"
			   ("SUSPECTED OR ACCUSED"
			      "SUSPECTED OR ACCUSED BY AUTHORITIES")
			   "POSSIBLE")
		      (xref perp-org-id))
  (phys-tgt-id        "12. PHYS TGT: ID" string)
  (phys-tgt-type      "13. PHYS TGT: TYPE" element
		      (set "CIVILIAN RESIDENCE"
			   "COMMERCIAL"
			   "COMMUNICATIONS"
			   "DIPLOMAT OFFICE OR RESIDENCE"
			   "ENERGY"
			   "FINANCIAL"
			   "LAW ENFORCEMENT FACILITY"
			   ("POLITICAL FIGURE OFFICE OR RESIDENCE"
			      "GOVERNMENT OFFICE OR RESIDENCE")
			   "ORGANIZATION OFFICE"
			   "TRANSPORT VEHICLE"
			   "TRANSPORTATION FACILITY"
			   "TRANSPORTATION ROUTE"
			   "WATER"
			   "OTHER")
		      (xref phys-tgt-id))
  (phys-tgt-num       "14. PHYS TGT: NUMBER" number
		      (xref phys-tgt-id))
  (phys-tgt-nation    "15. PHYS TGT: FOREIGN NATION" element
                      (set "AFGHANISTAN"
                           "ALBANIA"
                           "ANGOLA"
                           "ARGENTINA"
                           "AUSTRALIA"
                           "AUSTRIA"
                           "BAHAMAS"
                           "BANGLADESH"
                           "BARBADOS"
                           "BELGIUM"
                           "BELIZE"
                           "BOLIVIA"
                           "BRAZIL"
                           "BULGARIA"
                           "CAICOS ISLANDS"
                           "CANADA"
                           "CHILE"
                           "COLOMBIA"
                           "COSTA RICA"
                           "CUBA"
                           "CZECHOSLOVAKIA"
                           "DEMOCRATIC PEOPLES REP OF KOREA"
                           "DENMARK"
                           "DOMINICAN REPUBLIC"
                           "EAST GERMANY"
                           "ECUADOR"
                           "EGYPT"
                           "EL SALVADOR"
                           "FINLAND"
                           "FRANCE"
                           "GRENADA"
                           "GUATEMALA"
                           "HAITI"
                           "HONDURAS"
                           "HUNGARY"
                           "IRAN"
                           "IRAQ"
                           "ISRAEL"
                           "ITALY"
                           "JAMAICA"
                           "JAPAN"
                           "LEBANON"
                           "LIBYA"
                           "LUXEMBOURG"
                           "MALTA"
                           "MEXICO"
                           "MOROCCO"
                           "NETHERLANDS"
                           "NICARAGUA"
                           "NIGERIA"
                           "NORWAY"
                           "PAKISTAN"
                           "PANAMA"
                           "PARAGUAY"
                           "PEOPLES REP OF CHINA"
                           "PERU"
                           "PHILIPPINES"
                           "POLAND"
                           "REPUBLIC OF CHINA"
                           "REPUBLIC OF KOREA"
                           "REPUBLIC OF SOUTH AFRICA"
                           "ROMANIA"
                           "SPAIN"
                           "SWEDEN"
                           "SWITZERLAND"
                           "TRINIDAD AND TOBAGO"
                           "TURKS ISLANDS"
                           "UNITED KINGDOM"
                           "UNITED STATES"
                           "URUGUAY"
                           "USSR"
                           "VATICAN CITY"
                           "VENEZUELA"
                           "VIETNAM"
                           "WEST GERMANY"
                           "YUGOSLAVIA")
                      (xref phys-tgt-id))
  (phys-tgt-effect    "16. PHYS TGT: EFFECT OF INCIDENT" element
                      (set ("DESTROYED"
      			     "SOME DAMAGE") 
			   "NO DAMAGE"
                           "MONEY TAKEN FROM TARGET" 
			   "PROPERTY TAKEN FROM TARGET" 
			   "TARGET TAKEN")
                      (xref phys-tgt-id))
  (phys-tgt-total-num "17. PHYS TGT: TOTAL NUMBER" number)
  (hum-tgt-name       "18. HUM TGT: NAME" string)
  (hum-tgt-desc       "19. HUM TGT: DESCRIPTION" string
		      (xref hum-tgt-name))
  (hum-tgt-type       "20. HUM TGT: TYPE" element
                      (set "CIVILIAN"
                           "DIPLOMAT"
                           ("POLITICAL FIGURE"
			     ("GOVERNMENT OFFICIAL"
			       "FORMER GOVERNMENT OFFICIAL"))
                           "LEGAL OR JUDICIAL"
                           ("ACTIVE MILITARY"
			     "FORMER ACTIVE MILITARY")
                           "LAW ENFORCEMENT"
                           "SECURITY GUARD")
                      (xref hum-tgt-name hum-tgt-desc))
  (hum-tgt-num        "21. HUM TGT: NUMBER" number
		      (xref hum-tgt-name hum-tgt-desc))
  (hum-tgt-nation     "22. HUM TGT: FOREIGN NATION" element
                      (set "AFGHANISTAN"
                           "ALBANIA"
                           "ANGOLA"
                           "ARGENTINA"
                           "AUSTRALIA"
                           "AUSTRIA"
                           "BAHAMAS"
                           "BANGLADESH"
                           "BARBADOS"
                           "BELGIUM"
                           "BELIZE"
                           "BOLIVIA"
                           "BRAZIL"
                           "BULGARIA"
                           "CAICOS ISLANDS"
                           "CANADA"
                           "CHILE"
                           "COLOMBIA"
                           "COSTA RICA"
                           "CUBA"
                           "CZECHOSLOVAKIA"
                           "DEMOCRATIC PEOPLES REP OF KOREA"
                           "DENMARK"
                           "DOMINICAN REPUBLIC"
                           "EAST GERMANY"
                           "ECUADOR"
                           "EGYPT"
                           "EL SALVADOR"
                           "FINLAND"
                           "FRANCE"
                           "GRENADA"
                           "GUATEMALA"
                           "HAITI"
                           "HONDURAS"
                           "HUNGARY"
                           "IRAN"
                           "IRAQ"
                           "ISRAEL"
                           "ITALY"
                           "JAMAICA"
                           "JAPAN"
                           "LEBANON"
                           "LIBYA"
                           "LUXEMBOURG"
                           "MALTA"
                           "MEXICO"
                           "MOROCCO"
                           "NETHERLANDS"
                           "NICARAGUA"
                           "NIGERIA"
                           "NORWAY"
                           "PAKISTAN"
                           "PANAMA"
                           "PARAGUAY"
                           "PEOPLES REP OF CHINA"
                           "PERU"
                           "PHILIPPINES"
                           "POLAND"
                           "REPUBLIC OF CHINA"
                           "REPUBLIC OF KOREA"
                           "REPUBLIC OF SOUTH AFRICA"
                           "ROMANIA"
                           "SPAIN"
                           "SWEDEN"
                           "SWITZERLAND"
                           "TRINIDAD AND TOBAGO"
                           "TURKS ISLANDS"
                           "UNITED KINGDOM"
                           "UNITED STATES"
                           "URUGUAY"
                           "USSR"
                           "VATICAN CITY"
                           "VENEZUELA"
                           "VIETNAM"
                           "WEST GERMANY"
                           "YUGOSLAVIA")
                      (xref hum-tgt-name hum-tgt-desc))
  (hum-tgt-effect     "23. HUM TGT: EFFECT OF INCIDENT" element
                      (set "INJURY" 
			   "DEATH" 
			   ("NO INJURY"
			      ("NO DEATH"
			        "NO INJURY OR DEATH"))
                           "REGAINED FREEDOM" 
			   "ESCAPED"
                           "RESIGNATION" 
			   "NO RESIGNATION")
                      (xref hum-tgt-name hum-tgt-desc))
  (hum-tgt-total-num  "24. HUM TGT: TOTAL NUMBER" number))

