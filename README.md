# Soomi — Minimal Chess Engine (Go)

Soomi is a small, educational chess engine written in Go. It implements core chess-engine techniques (bitboards, negamax search, quiescence, TT, simple evaluation) as a learning project rather than a competitive engine.

## Features
- Bitboard move generation (now magic) 
- Evaluation: material, piece-square tables, mobility, passed pawns, king safety; middlegame and endgame interpolation (phase blend)  
- Move ordering: hash move, promotions, MVV-LVA captures, killers  
- Quiescence search with delta pruning 
- Transposition table (TT)  
- Negamax search with alpha-beta, late-move reductions (LMR), null move pruning (NMP), late move pruning (LMP) razoring, mate-distance pruning  
- UCI loop (works with Arena and other GUIs)  
- Iterative deepening with depth-based aspiration windows  
- Simple time control

## Limitations
- No FEN parsing (input/output limited)  
- No pondering / no multi-threading (single-threaded)  
- Basic move ordering and evaluation (hand-tuned)  
- Intended for learning and experimentation, not overly optimized for strength

## Build & Run
1. Install Go (1.20+ recommended, dont know combatibility with older versions): https://go.dev/doc/install  
2. Into terminal paste either:
3. go build -o Soomi.exe soomi.go (provides normal, about 2500kb)
4. go build -trimpath -ldflags "-s -w" -o Soomi.exe soomi.go (smaller, about 1700 kb)

## Current issues/bugs
1. LMP seems too aggressive, perhaps causing a lot of search instability for what its worth.
2. Passed pawns has an issue detecting if a pawn is passed if enemy has a pawn on the same rank. (FIXED)
3. Mate scores not printed when exiting on a forced mate found.
4. During fast time controls info is not printed for some reason.
5. Pv lines truncated, very ugly looking.
I am focused currently on fixing each one of these.

Any suggestions are welcome on how to improve the engine.
License is completely free to distribute as long as you mention the origin.

Thanks to:
- Maksim Korzh for the mention on his BBC (BitBoard Chess) engines github page: https://github.com/maksimKorzh/bbc?tab=readme-ov-file
- Chess programming wiki for awesome and plentiful information
