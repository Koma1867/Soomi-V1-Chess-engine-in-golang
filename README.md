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

- No fen parsing
- No pondering
- Single threaded
- Move ordering is basic
- Evaluation terms are hand picked
