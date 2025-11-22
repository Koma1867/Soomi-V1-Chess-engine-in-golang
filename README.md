# Soomi — Minimal Chess Engine (Go)

-Soomi is a small, educational chess engine written in Go. It started as a learning project to understand core chess programming techniques like bitboards and search algorithms.

-Despite this it is able to produce almost master level chess play, my estimations put it above 2000 elo.

## Features
- Bitboard move generation using Magic Bitboards
- Evaluation based on material, piece-square tables, mobility, passed pawns, and king safety
- Tapered evaluation to smoothly blend middlegame and endgame scores
- Negamax search with Alpha-Beta pruning
- Principal Variation Search (PVS) for improved search efficiency
- Iterative Deepening with depth-based Aspiration Windows
- Transposition Table to cache search results
- Quiescence Search with Delta Pruning to resolve tactical sequences
- Pruning techniques including Late Move Reductions (LMR), Null Move Pruning (NMP), Late Move Pruning (LMP), Razoring, and Mate Distance Pruning
- Move ordering using Hash move, MVV-LVA (captures), and Killers
- Full UCI protocol support (compatible with Arena, Banksia, and other GUIs)
- Simple time control management

## Limitations
- No FEN parsing: Input/output is limited to UCI position commands.
- Single-threaded: The engine runs on a single core and does not support pondering.
- Hand-tuned Evaluation: The evaluation terms are manually tuned rather than using automated tuning methods like SPSA.

## Build & Run
Install Go (1.20+ recommended): https://go.dev/doc/install

To build, run one of the following in your terminal:

```bash
# Standard build
go build -o Soomi.exe soomi.go

# Optimized/Smaller build (strips debug info)
go build -trimpath -ldflags "-s -w" -o Soomi.exe soomi.go
```

## Current Issues
- Insufficient Material: The engine doesn't automatically detect insufficient material draws (like King vs King), so it might continue playing in drawn positions.
- Move Generation: The move generator occasionally produces illegal pawn double pushes. These are correctly rejected by the legality check, so they don't affect gameplay, but it's a known quirk.

## License
Free to distribute and modify. Please credit the original author (Otto Laukkanen) if you use this code.

## Acknowledgments
- Maksim Korzh for the mention on his BBC (BitBoard Chess) engine's GitHub page.
- Chess Programming Wiki for the wealth of knowledge and algorithms.
