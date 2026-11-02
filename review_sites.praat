# Review participant site TextGrids, one file after another.
# Opens each participant wav with its <name>.TextGrid, else <name>.auto.TextGrid. "Save" writes
# <name>.TextGrid, which analyse_priming.py then uses instead of the projection.
# Adapted from Will Styler's textgrid_revisor.praat and textgrid_helperv3.praat
# (github.com/stylerw/styler_praat_scripts).

form Review participant site TextGrids
	sentence Corpus corpus
	sentence Results results
	comment Only files with check_alignment or an uncertain vowel in sites.csv:
	boolean Flagged_only 0
	comment Also reopen files that already have a hand-corrected TextGrid:
	boolean Redo_manual 0
	comment Off: only list the queue in the Info window.
	boolean Open_editor 1
endform

participants$ = corpus$ + "/participants/"
sites$ = results$ + "/sites.csv"
table = 0
if fileReadable (sites$)
	table = Read Table from comma-separated file: sites$
	nrows = Get number of rows
elsif flagged_only
	exitScript: "flagged_only needs ", sites$, ": run analyse_priming.py first."
endif

wavs = Create Strings as file list: "wavs", participants$ + "*.wav"
nfiles = Get number of strings
writeInfoLine: "file", tab$, "grid", tab$, "flags", tab$, "ceiling"
for ifile to nfiles
	selectObject: wavs
	wav$ = Get string: ifile
	name$ = wav$ - ".wav"
	manual$ = participants$ + name$ + ".TextGrid"
	auto$ = participants$ + name$ + ".auto.TextGrid"
	if fileReadable (manual$)
		grid$ = manual$
		source$ = "manual"
	elsif fileReadable (auto$)
		grid$ = auto$
		source$ = "auto"
	else
		grid$ = ""
	endif
	if grid$ <> "" and (source$ = "auto" or redo_manual)
		# <participant>_<sentence>_<version>, as in analyse_priming.py.
		u1 = index (name$, "_")
		participant$ = left$ (name$, u1 - 1)
		rest$ = mid$ (name$, u1 + 1, length (name$))
		u2 = index (rest$, "_")
		sentence$ = left$ (rest$, u2 - 1)
		heard$ = mid$ (rest$, u2 + 1, length (rest$))
		flags$ = ""
		ceiling = 5500
		if table
			selectObject: table
			for row to nrows
				p$ = Get value: row, "participant"
				s$ = Get value: row, "sentence"
				h$ = Get value: row, "heard"
				if p$ = participant$ and s$ = sentence$ and h$ = heard$
					c$ = Get value: row, "ceiling"
					if number (c$) <> undefined
						ceiling = number (c$)
					endif
					a$ = Get value: row, "check_alignment"
					if a$ = "True" and index (flags$, "align") = 0
						flags$ = flags$ + "align "
					endif
					unc$ = Get value: row, "uncertain"
					if unc$ = "True"
						site$ = Get value: row, "site"
						flags$ = flags$ + site$ + "? "
					endif
				endif
			endfor
		endif
		if flags$ <> "" or not flagged_only
			appendInfoLine: name$, tab$, source$, tab$, flags$, tab$, ceiling
			if open_editor
				call Review
				if clicked = 1
					goto done
				endif
			endif
		endif
	endif
endfor
label done
removeObject: wavs
if table
	removeObject: table
endif

procedure Review
	sound = Read from file: participants$ + wav$
	# Louder copy for listening to quiet recordings; the TextGrid is all that gets saved.
	Scale intensity: 70
	tg = Read from file: grid$
	plusObject: sound
	View & Edit
	editor: tg
		Show analyses: "yes", "no", "no", "yes", "no", 10
		# The speaker's ceiling from sites.csv, so the tracks match what was measured.
		Formant settings: ceiling, 5, 0.025, 30, 1
	endeditor
	beginPause: "Review " + name$
		comment: "Fix V/R/L boundaries; replace ? with the value heard (V1=E)."
		comment: "Flags: " + flags$
	clicked = endPause: "Stop", "Skip", "Save", 3
	if clicked = 3
		selectObject: tg
		Save as text file: manual$
	endif
	removeObject: sound, tg
endproc
