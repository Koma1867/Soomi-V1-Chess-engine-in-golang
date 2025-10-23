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

Tournament results:
Time control blitz 1 minute no increment, TSCP and Soomi no opening book
-----------------Hermann 2.8 64 bit-----------------
Hermann 2.8 64 bit - Nejmet 3.07   : 4,0/5 4-1-0 (01111)  80%  +241
Hermann 2.8 64 bit - Ruffian 1.0.5 : 2,0/5 2-3-0 (00101)  40%   -70
Hermann 2.8 64 bit - Soomi         : 4,5/5 4-0-1 (1111=)  90%  +382
Hermann 2.8 64 bit - Tscp181       : 5,0/5 5-0-0 (11111) 100% +1200
-----------------Nejmet 3.07-----------------
Nejmet 3.07 - Hermann 2.8 64 bit   : 1,0/5 1-4-0 (10000)  20%  -241
Nejmet 3.07 - Ruffian 1.0.5        : 1,0/5 1-4-0 (01000)  20%  -241
Nejmet 3.07 - Soomi                : 4,5/5 4-0-1 (11=11)  90%  +382
Nejmet 3.07 - Tscp181              : 4,0/5 4-1-0 (11110)  80%  +241
-----------------Ruffian 1.0.5-----------------
Ruffian 1.0.5 - Hermann 2.8 64 bit : 3,0/5 3-2-0 (11010)  60%   +70
Ruffian 1.0.5 - Nejmet 3.07        : 4,0/5 4-1-0 (10111)  80%  +241
Ruffian 1.0.5 - Soomi              : 5,0/5 5-0-0 (11111) 100% +1200
Ruffian 1.0.5 - Tscp181            : 5,0/5 5-0-0 (11111) 100% +1200
-----------------Soomi-----------------
Soomi - Hermann 2.8 64 bit         : 0,5/5 0-4-1 (0000=)  10%  -382
Soomi - Nejmet 3.07                : 0,5/5 0-4-1 (00=00)  10%  -382
Soomi - Ruffian 1.0.5              : 0,0/5 0-5-0 (00000)   0% -1200
Soomi - Tscp181                    : 5,0/5 5-0-0 (11111) 100% +1200
-----------------Tscp181-----------------
Tscp181 - Hermann 2.8 64 bit       : 0,0/5 0-5-0 (00000)   0% -1200
Tscp181 - Nejmet 3.07              : 1,0/5 1-4-0 (00001)  20%  -241
Tscp181 - Ruffian 1.0.5            : 0,0/5 0-5-0 (00000)   0% -1200
Tscp181 - Soomi                    : 0,0/5 0-5-0 (00000)   0% -1200
