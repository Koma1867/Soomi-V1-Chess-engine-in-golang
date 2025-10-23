# Soomi-V1-Chess-engine-in-golang
A chess engine written in golang, following basic chess programming principles from chess programming wiki.

Features:

- Move generation using bitboards

- Evaluation of material, piece square tables, piece mobility and passed pawns.
  Evaluation is interpolated between endgame and middlegame using fruits phase blending.

- Move ordering is simple of Hash move first, then promotions, captures sorted by MVV-LVA, and lastly killer moves.

- Quiescence features delta pruning, and capture only generation unless in check.

- Transposition table

- Negamax with Late Move Reductions, Razoring, Mate distance Pruning and Alpha Beta Pruning.

- Uci Loop to handle GUIs, Works atleast with Arena GUI.

- Search function to find best move using iterative deepening with depth based aspiration windows.

- Simple time control.

Limitations:

- No fen parsing.
- No pondering.
- Single threaded.
- Move ordering is basic.
- Evaluation terms are hand picked.

What made me build this engine and why in golang:

- Golang because its really the only coding language i know how to write, other than python.
- What made me write this was the fascination of original chess engines, strenght is not really that important to me, i value originality and doing something myself than copying from everyone else.
- But i do have to say the chess programming wiki has been a great help, without them i would have never got the engine this far, so thanks to everyone over there if they are reading this.

  To Note: Sometimes antivirus says the executable is a virus, if you want to use the file antivirus has to be switched off or you have to build the file yourself inside terminal.
  You have to first download golang from https://go.dev/doc/install and then in terminal navigate to where Soomi.go file is and type "go build -o Soomi.exe soomi.go" 
