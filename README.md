# Soomi Chess Engine (Go)

<p align="center">
  <img src="Soomi_logo.png" alt="Soomi Chess Engine Logo" width="1000">
</p>

<h1 align="center">Soomi Chess Engine</h1>

-Soomi is a small chess engine written in Go. It started as a learning project to understand core chess programming techniques like bitboards and search algorithms.

-Despite this it is able to produce almost master level chess play, my estimations put it above 2000 elo.

## Features
- Bitboard move generation using Magic Bitboards
- Evaluation based on material, piece-square tables, mobility, passed pawns, and king safety
- Tapered evaluation to smoothly blend middlegame and endgame scores
- Negamax search with Alpha-Beta pruning
- Principal Variation Search (PVS) for improved search efficiency
- Iterative Deepening with depth-based Aspiration Windows
- Transposition Table to cache search results
- Check extensions
- Quiescence Search with Delta Pruning to resolve tactical sequences
- Pruning techniques including Late Move Reductions (LMR), Null Move Pruning (NMP), Razoring, and Mate Distance Pruning
- Move ordering using Hash move, MVV-LVA (captures), and killers & piece square tables for quiet moves
- Full UCI protocol support (compatible with Arena, others not tested)
- Simple time management

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

# Disabling go bounds checking, perhaps 5% faster
go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi.exe soomi.go
```

## Current Issues
- Insufficient Material: The engine doesn't automatically detect insufficient material draws (like King vs King), so it might continue playing in drawn positions.

## License
Free to distribute and modify. Please credit the original author (Otto Laukkanen) if you use this code.

## Acknowledgments
- Maksim Korzh for the mention on his BBC (BitBoard Chess) engine's GitHub page.
- Chess Programming Wiki for the good information.
