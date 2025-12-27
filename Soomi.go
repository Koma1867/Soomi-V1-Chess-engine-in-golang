package main

import (
	"bufio"
	"fmt"
	"math/bits"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

const (
	White  = 0
	Black  = 1
	Pawn   = 0
	Knight = 1
	Bishop = 2
	Rook   = 3
	Queen  = 4
	King   = 5
)

/*
  ----------------------------------------------------------------------------------
   BITBOARD REPRESENTATION
  ----------------------------------------------------------------------------------
   The board is represented as a series of 64-bit integers, where each bit corresponds to a square on the chess board.

   Visual Mapping

   Rank 8 | 56 57 58 59 60 61 62 63
   Rank 7 | 48 49 50 51 52 53 54 55
   Rank 6 | 40 41 42 43 44 45 46 47
   Rank 5 | 32 33 34 35 36 37 38 39
   Rank 4 | 24 25 26 27 28 29 30 31
   Rank 3 | 16 17 18 19 20 21 22 23
   Rank 2 |  8  9 10 11 12 13 14 15
   Rank 1 |  0  1  2  3  4  5  6  7
           -----------------------
             A  B  C  D  E  F  G  H
*/

const (
	MaxDepth                           = 32
	Infinity                           = 30000
	Mate                               = 29000
	AspirationBase                     = 50
	AspirationStartDepth               = 4
	DefaultMovesToGo                   = 20
	NodeCheckMaskSearch                = 1023
	DeltaMargin                        = 150
	LMRMinChildDepth                   = 3
	LMRLateMoveAfter                   = 6
	MateScoreGuard                     = 1000
	defaultTTSizeMB                    = 256
	scoreHash                          = 1000000
	scorePromoBase                     = 900000
	scoreCaptureBase                   = 800000
	scoreKiller1                       = 750000
	scoreKiller2                       = 740000
	minTimeMs            int64         = 5
	perMoveCapDiv        int64         = 3
	nextIterMult                       = 2
	continueMargin       time.Duration = 10 * time.Millisecond
	MaxGamePly                         = 1024
	ZobristSeed                        = 1070372
	totalPhase                         = 24
)

const (
	FlagQuiet         = 0
	FlagCapture       = 4
	FlagEP            = 5
	FlagCastle        = 2
	FlagPromoN        = 8
	FlagPromoB        = 9
	FlagPromoR        = 10
	FlagPromoQ        = 11
	FlagPromoCN       = 12
	FlagPromoCB       = 13
	FlagPromoCR       = 14
	FlagPromoCQ       = 15
	ttFlagExact uint8 = 0
	ttFlagLower uint8 = 1
	ttFlagUpper uint8 = 2
)

const (
	PhaseScale   = 256
	MVVLVAWeight = 100
)

var (
	pieceValues       = [6]int{100, 320, 300, 500, 900, 20000}
	pst               [2][6][64]int
	pstEnd            [2][6][64]int
	piecePhase        = [6]int{0, 1, 1, 2, 4, 0}
	currentTC         atomic.Pointer[TimeControl]
	tt                *TranspositionTable
	rookMagics        [64]MagicEntry
	bishopMagics      [64]MagicEntry
	rookAttackTable   [102400]Bitboard
	bishopAttackTable [5248]Bitboard
)

/*
  ----------------------------------------------------------------------------------
   MAGIC BITBOARDS
  ----------------------------------------------------------------------------------
   Magic bitboards allows for instant sliding piece attacks

   The concept goes as follows
   1. Mask     A. Isolate relevant blocker squares for a piece at square X.
   2. Multiply B. Occupancy * MagicNumber (this scatters bits into high-order positions).
   3. Shift    C. Right shift to extract an index.
   4. Lookup   D. Use index to fetch pre-calculated attack set from a table.

   [ Occupancy on Board ]  &  [ Movement Mask ]
             |
             v
      [ Masked Blockers ]  * [ Magic Number ]
             |
             v
      [   Scattered Bits (Hash)   ]  >>  (64 - BitsInMask)
             |
             v
        [ Index ]  --->  [ Attack Table ]  --->  [ Attack Bitboard ]
*/

type MagicEntry struct {
	mask   Bitboard
	magic  Bitboard
	shift  uint8
	offset uint32
}

var rookMagicNumbers = [64]uint64{
	0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
	0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
	0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
	0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
	0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
	0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
	0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
	0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
	0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
	0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
	0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
	0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
	0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
	0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
	0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
	0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2,
}

var bishopMagicNumbers = [64]uint64{
	0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
	0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
	0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
	0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
	0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
	0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
	0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
	0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
	0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
	0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
	0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
	0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
	0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
	0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
	0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
	0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200,
}

/*
  ----------------------------------------------------------------------------------
   MOVE BIT-PACKING
  ----------------------------------------------------------------------------------
   To save memory and increase speed, a Move is not a struct, but a single 32-bit integer.
   We pack the 'From' square, 'To' square, and 'Flags' (promotion/capture type) into specific bits.

   [ 31 ... 16 ] [ 15 14 13 12 ] [ 11 10 9 8 7 6 ] [ 5 4 3 2 1 0 ]
     (Unused)       Flag (4b)       To Sq (6b)      From Sq (6b)
                       ^                 ^                ^
                       |                 |                |
   Example:           0100 (Capture)    111000 (H8)      000000 (A1)

   - Mask 0x3F (63) extracts squares (0-63).
   - Shift >> 6 moves the 'To' bits to the bottom.
   - Shift >> 12 moves the 'Flag' bits to the bottom.
*/

type Bitboard uint64
type Move uint32

type Position struct {
	pieces           [2][6]Bitboard
	occupied         [2]Bitboard
	all              Bitboard
	side             int
	castle           int
	epSquare         int
	halfmove         int
	hash             uint64
	material         [2]int
	psqScore         [2]int
	psqScoreEG       [2]int
	square           [64]int
	historyKeys      [MaxGamePly]uint64
	historyPly       int
	lastIrreversible int
	localNodes       int64
}

type Undo struct {
	castle           int
	epSquare         int
	halfmove         int
	captured         int
	lastIrreversible int
}

type TimeControl struct {
	wtime     int64
	btime     int64
	winc      int64
	binc      int64
	movestogo int
	movetime  int64
	infinite  bool
	depth     int
	deadline  time.Time
	stopped   int32
}

type SearchStack struct {
	killer1 Move
	killer2 Move
}

/*
  ----------------------------------------------------------------------------------
   TAPERED EVALUATION (Phase Calculation)
  ----------------------------------------------------------------------------------
   Chess strategy changes as pieces are exchanged. We use a "Phase" variable
   to interpolate between Middle Game (MG) and End Game (EG) scores.
   This combats evaluation discontinuity

   Total Phase = 24 (Based on piece weights: N=1, B=1, R=2, Q=4)

   Score Formula:
      FinalScore = ( (MG_Score * Phase) + (EG_Score * (256 - Phase)) ) / 256

      Start Position (all 32 pieces)       King vs King
      (Phase = 256)                       (Phase = 0)
      [========================================]
      ^                                        ^
      |                                        |
   Use mostly MG values                     Use mostly EG values
*/

func (p *Position) computePhase() int {
	rem := 0
	for pt := Knight; pt <= Queen; pt++ {
		bb := p.pieces[White][pt] | p.pieces[Black][pt]
		rem += bits.OnesCount64(uint64(bb)) * piecePhase[pt]
	}
	ph := totalPhase - rem
	if ph < 0 {
		return 0
	}
	return ph
}

func (p *Position) isEndgame() bool {
	phase := p.computePhase()
	return phase > 18
}

func rookMask(sq int) Bitboard {
	result := Bitboard(0)
	r, f := sq/8, sq%8

	for rr := r + 1; rr <= 6; rr++ {
		result |= Bitboard(1) << (rr*8 + f)
	}
	for rr := r - 1; rr >= 1; rr-- {
		result |= Bitboard(1) << (rr*8 + f)
	}

	for ff := f + 1; ff <= 6; ff++ {
		result |= Bitboard(1) << (r*8 + ff)
	}
	for ff := f - 1; ff >= 1; ff-- {
		result |= Bitboard(1) << (r*8 + ff)
	}

	return result
}

func bishopMask(sq int) Bitboard {
	result := Bitboard(0)
	r, f := sq/8, sq%8

	for rr, ff := r+1, f+1; rr <= 6 && ff <= 6; rr, ff = rr+1, ff+1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	for rr, ff := r-1, f+1; rr >= 1 && ff <= 6; rr, ff = rr-1, ff+1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	for rr, ff := r-1, f-1; rr >= 1 && ff >= 1; rr, ff = rr-1, ff-1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	for rr, ff := r+1, f-1; rr <= 6 && ff >= 1; rr, ff = rr+1, ff-1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	return result
}

func occupancyVariations(mask Bitboard) []Bitboard {
	bits := []int{}
	for sq := 0; sq < 64; sq++ {
		if mask&(Bitboard(1)<<sq) != 0 {
			bits = append(bits, sq)
		}
	}

	n := len(bits)
	count := 1 << n
	variations := make([]Bitboard, count)

	for i := 0; i < count; i++ {
		occ := Bitboard(0)
		for j := 0; j < n; j++ {
			if i&(1<<j) != 0 {
				occ |= Bitboard(1) << bits[j]
			}
		}
		variations[i] = occ
	}

	return variations
}

func magicIndex(occ Bitboard, magic Bitboard, shift uint8) uint32 {
	return uint32((occ * magic) >> shift)
}

func initMagicBitboards() {
	rookTableSize := 0
	bishopTableSize := 0

	for sq := 0; sq < 64; sq++ {
		mask := rookMask(sq)
		bitCount := bits.OnesCount64(uint64(mask))
		rookMagics[sq].mask = mask
		rookMagics[sq].magic = Bitboard(rookMagicNumbers[sq])
		rookMagics[sq].shift = 64 - uint8(bitCount)
		rookMagics[sq].offset = uint32(rookTableSize)
		rookTableSize += 1 << bitCount

		mask = bishopMask(sq)
		bitCount = bits.OnesCount64(uint64(mask))
		bishopMagics[sq].mask = mask
		bishopMagics[sq].magic = Bitboard(bishopMagicNumbers[sq])
		bishopMagics[sq].shift = 64 - uint8(bitCount)
		bishopMagics[sq].offset = uint32(bishopTableSize)
		bishopTableSize += 1 << bitCount
	}

	for sq := 0; sq < 64; sq++ {
		mask := rookMagics[sq].mask
		variations := occupancyVariations(mask)

		for _, occ := range variations {
			attacks := rookAttacksClassical(sq, occ)
			idx := magicIndex(occ, rookMagics[sq].magic, rookMagics[sq].shift)
			rookAttackTable[rookMagics[sq].offset+idx] = attacks
		}
	}

	for sq := 0; sq < 64; sq++ {
		mask := bishopMagics[sq].mask
		variations := occupancyVariations(mask)

		for _, occ := range variations {
			attacks := bishopAttacksClassical(sq, occ)
			idx := magicIndex(occ, bishopMagics[sq].magic, bishopMagics[sq].shift)
			bishopAttackTable[bishopMagics[sq].offset+idx] = attacks
		}
	}
}

func rookAttacksClassical(sq int, occ Bitboard) Bitboard {
	attacks := Bitboard(0)
	r, f := sq/8, sq%8

	for _, dr := range []int{1, -1} {
		for rr := r + dr; rr >= 0 && rr < 8; rr += dr {
			attacks |= Bitboard(1) << (rr*8 + f)
			if occ&(Bitboard(1)<<(rr*8+f)) != 0 {
				break
			}
		}
	}

	for _, df := range []int{1, -1} {
		for ff := f + df; ff >= 0 && ff < 8; ff += df {
			attacks |= Bitboard(1) << (r*8 + ff)
			if occ&(Bitboard(1)<<(r*8+ff)) != 0 {
				break
			}
		}
	}
	return attacks
}

func bishopAttacksClassical(sq int, occ Bitboard) Bitboard {
	attacks := Bitboard(0)
	r, f := sq/8, sq%8

	dirs := [][2]int{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
	for _, dir := range dirs {
		for rr, ff := r+dir[0], f+dir[1]; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr, ff = rr+dir[0], ff+dir[1] {
			attacks |= Bitboard(1) << (rr*8 + ff)
			if occ&(Bitboard(1)<<(rr*8+ff)) != 0 {
				break
			}
		}
	}
	return attacks
}

var sqBB [64]Bitboard

func initSqBB() {
	for i := 0; i < 64; i++ {
		sqBB[i] = Bitboard(1) << uint(i)
	}
}

var (
	zobristPiece    [2][6][64]uint64
	zobristSide     uint64
	zobristCastleWK uint64
	zobristCastleWQ uint64
	zobristCastleBK uint64
	zobristCastleBQ uint64
	zobristEP       [8]uint64
	knightAttacks   [64]Bitboard
	kingAttacks     [64]Bitboard
	pawnAttacks     [2][64]Bitboard
)

func (p *Position) isRepetition() bool {
	target := p.hash
	for i := p.historyPly - 2; i >= p.lastIrreversible; i -= 2 {
		if p.historyKeys[i] == target {
			return true
		}
	}
	return false
}

func (p *Position) isDraw() bool {
	return p.halfmove >= 100 || p.isRepetition()
}

type ttEntry struct {
	key    uint64
	packed uint64
}

func packEntry(move uint32, score int16, gen uint8, depth uint8, flag uint8) uint64 {
	return uint64(move)<<32 | uint64(uint16(score))<<16 | uint64(gen)<<8 | uint64(depth)<<2 | uint64(flag&0x3)
}

func (e ttEntry) unpack() (move uint32, score int16, gen uint8, depth uint8, flag uint8) {
	p := e.packed
	move = uint32(p >> 32)
	score = int16(p >> 16)
	gen = uint8(p >> 8)
	depth = uint8((p >> 2) & 0x3F)
	flag = uint8(p & 0x3)
	return
}

/*
  ----------------------------------------------------------------------------------
   TRANSPOSITION TABLE (TT)
  ----------------------------------------------------------------------------------
   A hash map that stores search results. It uses Zobrist Hashing, where the
   board state is XORed with random 64-bit numbers.

   Structure of an Entry (Packed into 64 bits):
   [    Move (32b)    ] [ Score (16b) ] [ Gen (8b) ] [ Depth (6b) ] [ Flag (2b) ]
   ^
   |
   (Upper 32 bits store the Move)

   Lookup Process:
   1. Compute Zobrist Hash of current position.
   2. Index = Hash % TableSize.
   3. Check if stored Key matches current Key.
   4. If Depth >= NeededDepth, use the stored Score immediately.
*/

type TranspositionTable struct {
	entries []ttEntry
	mask    uint64
	gen     uint32
}

func InitTT(sizeMB int) {
	if sizeMB <= 0 {
		sizeMB = defaultTTSizeMB
	}

	entrySize := uint64(16)
	totalBytes := uint64(sizeMB) * 1024 * 1024
	entries := totalBytes / entrySize
	size := uint64(1)
	for size<<1 <= entries {
		size <<= 1
	}
	maxLen := uint64(^uint(0) >> 1)
	if size > maxLen {
		size = maxLen
	}

	tt = &TranspositionTable{
		entries: make([]ttEntry, int(size)),
		mask:    size - 1,
		gen:     1,
	}
}

func (t *TranspositionTable) Clear() {
	t.gen++
	if t.gen > 255 {
		t.gen = 1
		for i := range t.entries {
			t.entries[i] = ttEntry{}
		}
	}
}

func (t *TranspositionTable) Probe(key uint64, minDepth int) (ttEntry, bool, bool) {
	idx := int(key & t.mask)
	e := t.entries[idx]
	if e.key != key {
		return ttEntry{}, false, false
	}
	_, _, gen, depth, _ := e.unpack()
	if gen != uint8(t.gen) {
		return e, true, false
	}
	return e, true, int(depth) >= minDepth
}

func (t *TranspositionTable) Save(key uint64, mv Move, score int, depth int, flag uint8) {
	depth = min(depth, 63)
	idx := int(key & t.mask)
	old := t.entries[idx]
	newPacked := packEntry(uint32(mv), int16(score), uint8(t.gen), uint8(depth), flag)
	if uint8(old.packed>>8) != uint8(t.gen) {
		t.entries[idx] = ttEntry{key: key, packed: newPacked}
		return
	}

	if old.key == key {
		oldDepth := uint8((old.packed >> 2) & 0x3F)
		oldFlag := uint8(old.packed & 0x3)
		if depth > int(oldDepth) || (depth == int(oldDepth) && flag == ttFlagExact && oldFlag != ttFlagExact) {
			t.entries[idx] = ttEntry{key: key, packed: newPacked}
		}
	} else {
		oldDepth := uint8((old.packed >> 2) & 0x3F)
		if depth >= int(oldDepth) {
			t.entries[idx] = ttEntry{key: key, packed: newPacked}
		}
	}
}

func init() {
	initPST()
	initZobrist()
	initSqBB()
	initAttacks()
	initMagicBitboards()
	InitTT(defaultTTSizeMB)
}

func initPST() {
	pst[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		-5, 10, 10, -20, -20, 10, 10, -5,
		0, 0, -10, 5, 5, 0, 0, 0,
		0, -10, 10, 20, 20, 10, 5, 0,
		10, 10, 15, 25, 25, 15, 10, 10,
		15, 15, 20, 30, 30, 20, 15, 15,
		30, 30, 30, 40, 40, 30, 30, 30,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pst[White][Knight] = [64]int{
		-30, -20, -10, -10, -10, -10, -20, -30,
		-20, -10, 5, 5, 5, 5, -10, -20,
		-20, 5, 15, 15, 15, 15, 5, -20,
		-10, 5, 15, 20, 20, 15, 5, -10,
		-10, 5, 15, 25, 25, 15, 5, -10,
		-20, 5, 10, 15, 15, 10, 5, -20,
		-20, 0, 0, 0, 0, 0, 0, -20,
		-30, -10, -10, -10, -10, -10, -20, -30,
	}

	pst[White][Bishop] = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 10, 5, 5, 5, 5, 10, -10,
		-10, 5, 5, 15, 15, 5, 5, -10,
		-10, 5, 5, 15, 15, 5, 5, -10,
		-10, 5, 10, 20, 20, 10, 5, -10,
		-10, 10, 10, 15, 15, 10, 10, -10,
		-10, 10, 5, 5, 5, 5, 10, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}

	pst[White][Rook] = [64]int{
		0, 0, 5, 10, 10, 5, 0, 0,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		10, 15, 15, 20, 20, 15, 15, 10,
		0, 0, 0, 5, 5, 0, 0, 0,
	}

	pst[White][Queen] = [64]int{
		-20, -10, -10, -5, -5, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 5, 5, 5, 5, 0, -10,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-10, 5, 5, 5, 5, 5, 5, -10,
		-10, 0, 5, 5, 5, 5, 0, -10,
		-20, -10, -10, -5, -5, -10, -10, -20,
	}

	pst[White][King] = [64]int{
		30, 20, 5, -10, -10, 5, 20, 30,
		10, 10, -15, -30, -30, -15, 10, 10,
		-20, -20, -20, -20, -20, -20, -20, -20,
		-20, -30, -30, -40, -40, -30, -30, -20,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
	}

	pstEnd[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		10, 10, 10, 10, 10, 10, 10, 10,
		20, 20, 20, 20, 20, 20, 20, 20,
		30, 30, 30, 30, 30, 30, 30, 30,
		40, 40, 40, 40, 40, 40, 40, 40,
		60, 60, 60, 60, 60, 60, 60, 60,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pstEnd[White][Knight] = [64]int{
		-20, -10, -5, -5, -5, -5, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 5, 5, 5, 5, 5, 5, -10,
		-5, 5, 5, 10, 10, 5, 5, -5,
		-5, 5, 5, 10, 10, 5, 5, -5,
		-10, 5, 5, 5, 5, 5, 5, -10,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-20, -10, -5, -5, -5, -5, -10, -20,
	}

	pstEnd[White][Bishop] = [64]int{
		-10, -5, -5, -5, -5, -5, -5, -10,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-10, -5, -5, -5, -5, -5, -5, -10,
	}

	pstEnd[White][Rook] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		15, 20, 20, 25, 25, 20, 20, 15,
		10, 10, 10, 10, 10, 10, 10, 10,
	}

	pstEnd[White][Queen] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pstEnd[White][King] = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 10, 20, 20, 10, 0, -10,
		-10, 0, 10, 30, 30, 10, 0, -10,
		-10, 0, 10, 30, 30, 10, 0, -10,
		-10, 0, 10, 20, 20, 10, 0, -10,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}

	for pt := 0; pt < 6; pt++ {
		for sq := 0; sq < 64; sq++ {
			bsq := sq ^ 56
			pst[Black][pt][sq] = pst[White][pt][bsq]
			pstEnd[Black][pt][sq] = pstEnd[White][pt][bsq]
		}
	}
}

/*
  ----------------------------------------------------------------------------------
   ZOBRIST HASHING INITIALIZATION
  ----------------------------------------------------------------------------------
   We assign a random 64-bit number to every possible board state component.

   Components Hashed:
   1. Piece at Square (e.g., White Pawn on E4)
   2. Side to Move (White/Black)
   3. Castling Rights (KQkq)
   4. En Passant File

   The final Board Hash is the XOR sum of all active components.
   Hash = [WP_on_E4] ^ [BK_on_E8] ^ [WhiteToMove] ^ ...

   Incremental Update:
   When a piece moves, we don't recalculate from scratch. We XOR out the old
   piece and XOR in the new one.
*/

func initZobrist() {
	rng := uint64(ZobristSeed)
	next := func() uint64 {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		return rng
	}

	for c := 0; c < 2; c++ {
		for pt := 0; pt < 6; pt++ {
			for sq := 0; sq < 64; sq++ {
				zobristPiece[c][pt][sq] = next()
			}
		}
	}
	zobristSide = next()
	zobristCastleWK = next()
	zobristCastleWQ = next()
	zobristCastleBK = next()
	zobristCastleBQ = next()
	for i := 0; i < 8; i++ {
		zobristEP[i] = next()
	}
}

func initAttacks() {
	knightDirs := []int{-17, -15, -10, -6, 6, 10, 15, 17}
	kingDirs := []int{-9, -8, -7, -1, 1, 7, 8, 9}

	for sq := 0; sq < 64; sq++ {
		r, f := sq>>3, sq&7

		for _, d := range knightDirs {
			to := sq + d
			if to >= 0 && to < 64 {
				tr, tf := to>>3, to&7
				if abs(r-tr) <= 2 && abs(f-tf) <= 2 {
					knightAttacks[sq] |= sqBB[to]
				}
			}
		}

		for _, d := range kingDirs {
			to := sq + d
			if to >= 0 && to < 64 {
				tr, tf := to>>3, to&7
				if abs(r-tr) <= 1 && abs(f-tf) <= 1 {
					kingAttacks[sq] |= sqBB[to]
				}
			}
		}

		if r < 7 && f > 0 {
			pawnAttacks[White][sq] |= sqBB[sq+7]
		}
		if r < 7 && f < 7 {
			pawnAttacks[White][sq] |= sqBB[sq+9]
		}
		if r > 0 && f > 0 {
			pawnAttacks[Black][sq] |= sqBB[sq-9]
		}
		if r > 0 && f < 7 {
			pawnAttacks[Black][sq] |= sqBB[sq-7]
		}
	}
}

func popLSB(b *Bitboard) int {
	idx := bits.TrailingZeros64(uint64(*b))
	*b &= *b - 1
	return idx
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func makeMoves(from, to, flags int) Move {
	return Move(from | (to << 6) | (flags << 12))
}

func (m Move) from() int       { return int(m) & 63 }
func (m Move) to() int         { return int(m>>6) & 63 }
func (m Move) flags() int      { return int(m >> 12) }
func (m Move) isCapture() bool { return m.flags()&4 != 0 }
func (m Move) isPromo() bool   { return m.flags()&8 != 0 }
func (m Move) promoType() int  { return (m.flags() & 3) + Knight }

func (m Move) String() string {
	if m == 0 {
		return "0000"
	}
	var buf [5]byte
	from, to := m.from(), m.to()
	buf[0] = byte('a' + from%8)
	buf[1] = byte('1' + from/8)
	buf[2] = byte('a' + to%8)
	buf[3] = byte('1' + to/8)
	if m.isPromo() {
		buf[4] = "nbrq"[m.promoType()-Knight]
		return string(buf[:5])
	}
	return string(buf[:4])
}

func NewPosition() *Position {
	p := &Position{}
	p.setStartPos()
	return p
}

func (p *Position) setStartPos() {
	*p = Position{}
	for i := range p.square {
		p.square[i] = -1
	}

	p.side = White
	p.castle = 0xF
	p.epSquare = -1

	setPiece := func(sq, c, pc int) {
		bb := sqBB[sq]
		p.square[sq] = (c << 3) | pc
		p.pieces[c][pc] |= bb
		p.occupied[c] |= bb
		p.all |= bb
		p.material[c] += pieceValues[pc]
		p.psqScore[c] += pst[c][pc][sq]
		p.psqScoreEG[c] += pstEnd[c][pc][sq]
		p.hash ^= zobristPiece[c][pc][sq]
	}

	for f := 0; f < 8; f++ {
		setPiece(8+f, White, Pawn)
	}
	setPiece(0, White, Rook)
	setPiece(7, White, Rook)
	setPiece(1, White, Knight)
	setPiece(6, White, Knight)
	setPiece(2, White, Bishop)
	setPiece(5, White, Bishop)
	setPiece(3, White, Queen)
	setPiece(4, White, King)

	for f := 0; f < 8; f++ {
		setPiece(48+f, Black, Pawn)
	}
	setPiece(56, Black, Rook)
	setPiece(63, Black, Rook)
	setPiece(57, Black, Knight)
	setPiece(62, Black, Knight)
	setPiece(58, Black, Bishop)
	setPiece(61, Black, Bishop)
	setPiece(59, Black, Queen)
	setPiece(60, Black, King)
	p.hash ^= zobristCastleWK
	p.hash ^= zobristCastleWQ
	p.hash ^= zobristCastleBK
	p.hash ^= zobristCastleBQ
	p.historyKeys[0] = p.hash
}

func (p *Position) pieceAt(sq int) (color, piece int, ok bool) {
	val := p.square[sq]
	if val < 0 {
		return 0, 0, false
	}
	color = val >> 3
	piece = val & 7
	return color, piece, true
}

func rookAttacks(sq int, occ Bitboard) Bitboard {
	m := &rookMagics[sq]
	idx := uint32(((occ & m.mask) * m.magic) >> m.shift)
	return rookAttackTable[m.offset+idx]
}

func bishopAttacks(sq int, occ Bitboard) Bitboard {
	m := &bishopMagics[sq]
	idx := uint32(((occ & m.mask) * m.magic) >> m.shift)
	return bishopAttackTable[m.offset+idx]
}

func (p *Position) isAttacked(sq, bySide int) bool {
	if pawnAttacks[bySide^1][sq]&p.pieces[bySide][Pawn] != 0 {
		return true
	}
	if knightAttacks[sq]&p.pieces[bySide][Knight] != 0 {
		return true
	}
	if kingAttacks[sq]&p.pieces[bySide][King] != 0 {
		return true
	}
	qu := p.pieces[bySide][Queen]
	if bishopAttacks(sq, p.all)&(p.pieces[bySide][Bishop]|qu) != 0 {
		return true
	}
	return rookAttacks(sq, p.all)&(p.pieces[bySide][Rook]|qu) != 0
}

func (p *Position) inCheck() bool {
	kingBB := p.pieces[p.side][King]
	kingSq := bits.TrailingZeros64(uint64(kingBB))
	return p.isAttacked(kingSq, p.side^1)
}

/*
   Move Generation Strategy:
   Instead of looping over every square, we loop over pieces (Bitboards).

   Example:
   for bb := knights; bb != 0; {
       from := popLSB(&bb)         // Get location of a knight
       attacks := lookup(from)     // Get all target squares
       valid := attacks & ~us      // Remove friendly fire
   }

   This _Piece-Centric_ approach is much faster than _Square-Centric_ loops
   because empty squares are skipped entirely.
*/

func (p *Position) generateMovesTo(buf []Move, capturesOnly bool) int {
	i := 0
	us, them := p.side, p.side^1

	occAll := p.all
	occUs := p.occupied[us]
	occThem := p.occupied[them]
	ep := p.epSquare
	pawns := p.pieces[us][Pawn]
	var push, dblPush int
	var promoRank, dblRank int
	if us == White {
		push, dblPush = 8, 16
		promoRank, dblRank = 6, 1
	} else {
		push, dblPush = -8, -16
		promoRank, dblRank = 1, 6
	}

	for bb := pawns; bb != 0; {
		from := popLSB(&bb)
		to := from + push

		if from>>3 == promoRank {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				buf[i] = makeMoves(from, to, FlagPromoQ)
				i++
				buf[i] = makeMoves(from, to, FlagPromoR)
				i++
				buf[i] = makeMoves(from, to, FlagPromoB)
				i++
				buf[i] = makeMoves(from, to, FlagPromoN)
				i++
			}

			attacks := pawnAttacks[us][from] & occThem
			for att := attacks; att != 0; {
				to := popLSB(&att)
				buf[i] = makeMoves(from, to, FlagPromoCQ)
				i++
				buf[i] = makeMoves(from, to, FlagPromoCR)
				i++
				buf[i] = makeMoves(from, to, FlagPromoCB)
				i++
				buf[i] = makeMoves(from, to, FlagPromoCN)
				i++
			}
		} else {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				buf[i] = makeMoves(from, to, FlagQuiet)
				i++

				if from>>3 == dblRank {
					to2 := from + dblPush
					if occAll&sqBB[to2] == 0 {
						buf[i] = makeMoves(from, to2, FlagQuiet)
						i++
					}
				}
			}

			attacks := pawnAttacks[us][from] & occThem
			for att := attacks; att != 0; {
				to := popLSB(&att)
				buf[i] = makeMoves(from, to, FlagCapture)
				i++
			}

			if ep >= 0 {
				if pawnAttacks[us][from]&sqBB[ep] != 0 {
					buf[i] = makeMoves(from, ep, FlagEP)
					i++
				}
			}
		}
	}

	for bb := p.pieces[us][Knight]; bb != 0; {
		from := popLSB(&bb)
		attacks := knightAttacks[from] & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	for bb := p.pieces[us][Bishop]; bb != 0; {
		from := popLSB(&bb)
		attacks := bishopAttacks(from, occAll) & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	for bb := p.pieces[us][Rook]; bb != 0; {
		from := popLSB(&bb)
		attacks := rookAttacks(from, occAll) & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	for bb := p.pieces[us][Queen]; bb != 0; {
		from := popLSB(&bb)
		attacks := (bishopAttacks(from, occAll) | rookAttacks(from, occAll)) & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	kingSq := bits.TrailingZeros64(uint64(p.pieces[us][King]))
	attacks := kingAttacks[kingSq] & ^occUs
	if capturesOnly {
		attacks &= occThem
	}
	for att := attacks; att != 0; {
		to := popLSB(&att)
		flag := FlagQuiet
		if occThem&sqBB[to] != 0 {
			flag = FlagCapture
		}
		buf[i] = makeMoves(kingSq, to, flag)
		i++
	}

	if !capturesOnly && !p.inCheck() {
		if us == White {
			if p.castle&1 != 0 && occAll&0x60 == 0 {
				buf[i] = makeMoves(4, 6, FlagCastle)
				i++
			}
			if p.castle&2 != 0 && occAll&0x0E == 0 {
				buf[i] = makeMoves(4, 2, FlagCastle)
				i++
			}
		} else {
			if p.castle&4 != 0 && occAll&(0x60<<56) == 0 {
				buf[i] = makeMoves(60, 62, FlagCastle)
				i++
			}
			if p.castle&8 != 0 && occAll&(0x0E<<56) == 0 {
				buf[i] = makeMoves(60, 58, FlagCastle)
				i++
			}
		}
	}
	return i
}

func (p *Position) isLegal(m Move) bool {
	from, to := m.from(), m.to()
	us, them := p.side, p.side^1
	flags := m.flags()
	if flags == FlagCastle {
		if to > from {
			if p.isAttacked(from+1, them) || p.isAttacked(to, them) {
				return false
			}
		} else {
			if p.isAttacked(from-1, them) || p.isAttacked(to, them) {
				return false
			}
		}
		return true
	}
	pt := p.square[from] & 7
	fromBB, toBB := sqBB[from], sqBB[to]
	theirP, theirN, theirB, theirR, theirQ, theirK := p.pieces[them][Pawn], p.pieces[them][Knight], p.pieces[them][Bishop], p.pieces[them][Rook], p.pieces[them][Queen], p.pieces[them][King]

	if m.isCapture() {
		capSq := to
		if flags == FlagEP {
			if us == White {
				capSq = to - 8
			} else {
				capSq = to + 8
			}
		}
		capBB := sqBB[capSq]
		if theirP&capBB != 0 {
			theirP &^= capBB
		} else if theirN&capBB != 0 {
			theirN &^= capBB
		} else if theirB&capBB != 0 {
			theirB &^= capBB
		} else if theirR&capBB != 0 {
			theirR &^= capBB
		} else {
			theirQ &^= capBB
		}
	}

	occ2 := p.all&^fromBB | toBB
	if flags == FlagEP {
		if us == White {
			occ2 &^= sqBB[to-8]
		} else {
			occ2 &^= sqBB[to+8]
		}
	}
	var kingSq int
	if pt == King {
		kingSq = to
	} else {
		kingSq = bits.TrailingZeros64(uint64(p.pieces[us][King]))
	}

	if pawnAttacks[them^1][kingSq]&theirP != 0 {
		return false
	}
	if knightAttacks[kingSq]&theirN != 0 {
		return false
	}
	if bishopAttacks(kingSq, occ2)&(theirB|theirQ) != 0 {
		return false
	}
	if rookAttacks(kingSq, occ2)&(theirR|theirQ) != 0 {
		return false
	}
	if kingAttacks[kingSq]&theirK != 0 {
		return false
	}
	return true
}

func (p *Position) makeMove(m Move) Undo {
	undo := Undo{
		castle:           p.castle,
		epSquare:         p.epSquare,
		halfmove:         p.halfmove,
		captured:         -1,
		lastIrreversible: p.lastIrreversible,
	}

	from, to, flags := m.from(), m.to(), m.flags()
	us, them := p.side, p.side^1
	h := p.hash
	h ^= zobristSide

	if p.epSquare >= 0 {
		h ^= zobristEP[p.epSquare%8]
		p.epSquare = -1
	}

	movingPiece := p.square[from] & 7

	if flags&FlagCapture != 0 {
		capSq := to
		if flags == FlagEP {
			if us == White {
				capSq = to - 8
			} else {
				capSq = to + 8
			}
		}
		capturedPiece := p.square[capSq] & 7
		undo.captured = capturedPiece

		if capturedPiece == Rook {
			switch capSq {
			case 0:
				p.castle &^= 2
			case 7:
				p.castle &^= 1
			case 56:
				p.castle &^= 8
			case 63:
				p.castle &^= 4
			}
		}

		bb := sqBB[capSq]
		p.pieces[them][capturedPiece] &^= bb
		p.occupied[them] &^= bb
		p.all &^= bb
		h ^= zobristPiece[them][capturedPiece][capSq]
		p.material[them] -= pieceValues[capturedPiece]
		p.psqScore[them] -= pst[them][capturedPiece][capSq]
		p.psqScoreEG[them] -= pstEnd[them][capturedPiece][capSq]
		p.square[capSq] = -1

		p.halfmove = 0
	} else if movingPiece == Pawn {
		p.halfmove = 0
	} else {
		p.halfmove++
	}

	if flags == FlagCastle {
		bbFrom := sqBB[from]
		bbTo := sqBB[to]
		p.pieces[us][King] &^= bbFrom
		p.pieces[us][King] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
		h ^= zobristPiece[us][King][from] ^ zobristPiece[us][King][to]
		p.psqScore[us] += pst[us][King][to] - pst[us][King][from]
		p.psqScoreEG[us] += pstEnd[us][King][to] - pstEnd[us][King][from]
		p.square[from] = -1
		p.square[to] = (us << 3) | King
		var rf, rt int
		if to > from {
			rf, rt = from+3, from+1
		} else {
			rf, rt = from-4, from-1
		}
		rfBB := sqBB[rf]
		rtBB := sqBB[rt]
		p.pieces[us][Rook] &^= rfBB
		p.pieces[us][Rook] |= rtBB
		p.occupied[us] &^= rfBB
		p.occupied[us] |= rtBB
		p.all &^= rfBB
		p.all |= rtBB
		h ^= zobristPiece[us][Rook][rf] ^ zobristPiece[us][Rook][rt]
		p.psqScore[us] += pst[us][Rook][rt] - pst[us][Rook][rf]
		p.psqScoreEG[us] += pstEnd[us][Rook][rt] - pstEnd[us][Rook][rf]
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		bbFrom := sqBB[from]
		p.pieces[us][Pawn] &^= bbFrom
		p.occupied[us] &^= bbFrom
		p.all &^= bbFrom
		h ^= zobristPiece[us][Pawn][from]
		p.material[us] -= pieceValues[Pawn]
		p.psqScore[us] -= pst[us][Pawn][from]
		p.psqScoreEG[us] -= pstEnd[us][Pawn][from]
		p.square[from] = -1

		promoType := (flags & 3) + Knight
		bbTo := sqBB[to]
		p.pieces[us][promoType] |= bbTo
		p.occupied[us] |= bbTo
		p.all |= bbTo
		h ^= zobristPiece[us][promoType][to]
		p.material[us] += pieceValues[promoType]
		p.psqScore[us] += pst[us][promoType][to]
		p.psqScoreEG[us] += pstEnd[us][promoType][to]
		p.square[to] = (us << 3) | promoType

	} else {
		bbFrom := sqBB[from]
		bbTo := sqBB[to]
		p.pieces[us][movingPiece] &^= bbFrom
		p.pieces[us][movingPiece] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
		h ^= zobristPiece[us][movingPiece][from] ^ zobristPiece[us][movingPiece][to]
		p.psqScore[us] += pst[us][movingPiece][to] - pst[us][movingPiece][from]
		p.psqScoreEG[us] += pstEnd[us][movingPiece][to] - pstEnd[us][movingPiece][from]
		p.square[from] = -1
		p.square[to] = (us << 3) | movingPiece

		if movingPiece == Pawn && abs(to-from) == 16 {
			p.epSquare = (from + to) / 2
			h ^= zobristEP[p.epSquare%8]
		}
	}

	switch movingPiece {
	case King:
		if us == White {
			p.castle &^= 3
		} else {
			p.castle &^= 12
		}
	case Rook:
		switch from {
		case 0:
			p.castle &^= 2
		case 7:
			p.castle &^= 1
		case 56:
			p.castle &^= 8
		case 63:
			p.castle &^= 4
		}
	}

	changedCastle := undo.castle ^ p.castle
	if changedCastle&1 != 0 {
		h ^= zobristCastleWK
	}
	if changedCastle&2 != 0 {
		h ^= zobristCastleWQ
	}
	if changedCastle&4 != 0 {
		h ^= zobristCastleBK
	}
	if changedCastle&8 != 0 {
		h ^= zobristCastleBQ
	}
	p.side ^= 1
	irreversible := (undo.captured >= 0) || (movingPiece == Pawn)
	p.historyPly++
	p.historyKeys[p.historyPly] = h

	if irreversible {
		p.lastIrreversible = p.historyPly
	}
	p.hash = h
	return undo
}

func (p *Position) unmakeMove(m Move, undo Undo) {
	from, to, flags := m.from(), m.to(), m.flags()
	us := p.side ^ 1
	them := p.side
	p.historyPly--
	p.lastIrreversible = undo.lastIrreversible
	p.side = us
	p.castle = undo.castle
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove

	if flags == FlagCastle {
		bbFrom := sqBB[to]
		bbTo := sqBB[from]
		p.pieces[us][King] &^= bbFrom
		p.pieces[us][King] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
		p.psqScore[us] += pst[us][King][from] - pst[us][King][to]
		p.psqScoreEG[us] += pstEnd[us][King][from] - pstEnd[us][King][to]
		p.square[to] = -1
		p.square[from] = (us << 3) | King

		var rf, rt int
		if to > from {
			rf, rt = from+1, from+3
		} else {
			rf, rt = from-1, from-4
		}

		rfBB := sqBB[rf]
		rtBB := sqBB[rt]
		p.pieces[us][Rook] &^= rfBB
		p.pieces[us][Rook] |= rtBB
		p.occupied[us] &^= rfBB
		p.occupied[us] |= rtBB
		p.all &^= rfBB
		p.all |= rtBB
		p.psqScore[us] += pst[us][Rook][rt] - pst[us][Rook][rf]
		p.psqScoreEG[us] += pstEnd[us][Rook][rt] - pstEnd[us][Rook][rf]
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		promoType := (flags & 3) + Knight
		bbTo := sqBB[to]
		p.pieces[us][promoType] &^= bbTo
		p.occupied[us] &^= bbTo
		p.all &^= bbTo
		p.material[us] -= pieceValues[promoType]
		p.psqScore[us] -= pst[us][promoType][to]
		p.psqScoreEG[us] -= pstEnd[us][promoType][to]
		p.square[to] = -1
		bbFrom := sqBB[from]
		p.pieces[us][Pawn] |= bbFrom
		p.occupied[us] |= bbFrom
		p.all |= bbFrom
		p.material[us] += pieceValues[Pawn]
		p.psqScore[us] += pst[us][Pawn][from]
		p.psqScoreEG[us] += pstEnd[us][Pawn][from]
		p.square[from] = (us << 3) | Pawn

	} else {
		movingPt := p.square[to] & 7
		bbFrom := sqBB[to]
		bbTo := sqBB[from]
		p.pieces[us][movingPt] &^= bbFrom
		p.pieces[us][movingPt] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
		p.psqScore[us] += pst[us][movingPt][from] - pst[us][movingPt][to]
		p.psqScoreEG[us] += pstEnd[us][movingPt][from] - pstEnd[us][movingPt][to]
		p.square[to] = -1
		p.square[from] = (us << 3) | movingPt
	}

	if undo.captured >= 0 {
		capSq := to
		if flags == FlagEP {
			if us == White {
				capSq = to - 8
			} else {
				capSq = to + 8
			}
		}
		bb := sqBB[capSq]
		p.pieces[them][undo.captured] |= bb
		p.occupied[them] |= bb
		p.all |= bb
		p.material[them] += pieceValues[undo.captured]
		p.psqScore[them] += pst[them][undo.captured][capSq]
		p.psqScoreEG[them] += pstEnd[them][undo.captured][capSq]
		p.square[capSq] = (them << 3) | undo.captured
	}
	p.hash = p.historyKeys[p.historyPly]
}

func (p *Position) makeNullMove() Undo {
	undo := Undo{
		epSquare: p.epSquare,
		halfmove: p.halfmove,
	}

	p.side ^= 1
	p.hash ^= zobristSide

	if epFile := p.epSquare; epFile != -1 {
		p.hash ^= zobristEP[epFile%8]
		p.epSquare = -1
	}
	p.halfmove++
	p.historyPly++
	p.historyKeys[p.historyPly] = p.hash
	return undo
}

func (p *Position) unmakeNullMove(undo Undo) {
	p.historyPly--
	p.hash = p.historyKeys[p.historyPly]
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove
	p.side ^= 1
}

func (p *Position) evaluate() int {
	a := p.material[White] - p.material[Black]
	b := p.psqScore[White] - p.psqScore[Black]
	c := p.psqScoreEG[White] - p.psqScoreEG[Black]

	mg := a + b
	eg := a + c

	ph := p.computePhase()
	phaseScaled := ((totalPhase-ph)*PhaseScale + totalPhase/2) / totalPhase
	score := eg + ((mg-eg)*phaseScaled)/PhaseScale

	if p.side == Black {
		return -score
	}
	return score
}

/*
  ----------------------------------------------------------------------------------
   MOVE ORDERING
  ----------------------------------------------------------------------------------
   To prune the search tree effectively, we must search the best moves first.

   Sorting Priority List:
   1. Hash Move     --> The best move found at the previous depth
   2. Promotions    --> Moves that create a Queen are usually critical
   3. Good Captures --> Most Valuable Victim, Least Valuable Attacker, e.g. Pawn(1) capturing Queen(9) is prioritized over Queen(9) capturing Pawn(1).
   4. Killer Moves  --> Quiet moves that caused a cutoff in a sibling node
   5. PST moves     --> Moves that according to PST move from a bad square to a better one.
*/

func (p *Position) orderMoves(moves []Move, bestMove, killer1, killer2 Move) []Move {
	n := len(moves)
	var stackScores [256]int
	scores := stackScores[:n]

	for i := 0; i < n; i++ {
		m := moves[i]
		score := 0
		if m == bestMove {
			score = scoreHash
		} else {
			if m.isPromo() {
				score = scorePromoBase + pieceValues[m.promoType()]
			} else if m.isCapture() {
				from := m.from()
				to := m.to()
				capSq := to
				if m.flags() == FlagEP {
					if p.side == White {
						capSq = to - 8
					} else {
						capSq = to + 8
					}
				}
				attacker := p.square[from] & 7
				victim := p.square[capSq] & 7
				score = scoreCaptureBase + pieceValues[victim]*MVVLVAWeight - pieceValues[attacker]
			} else {
				switch m {
				case killer1:
					score = scoreKiller1
				case killer2:
					score = scoreKiller2
				default:
					piece := p.square[m.from()] & 7
					score = pst[p.side][piece][m.to()] - pst[p.side][piece][m.from()]
				}
			}
		}
		scores[i] = score
	}

	for i := 1; i < n; i++ {
		kMove := moves[i]
		kScore := scores[i]
		j := i - 1
		for j >= 0 && scores[j] < kScore {
			moves[j+1] = moves[j]
			scores[j+1] = scores[j]
			j--
		}
		moves[j+1] = kMove
		scores[j+1] = kScore
	}

	return moves
}

/*
  ----------------------------------------------------------------------------------
   QUIESCENCE SEARCH
  ----------------------------------------------------------------------------------
   Standard search stops at a fixed depth, but this can lead to the "Horizon Effect",
   where the engine misses critical tactical sequences just beyond the search depth.

   Scenario:
   Depth 4: White takes Black Queen. Eval says White is +900. STOP (time ran out etc...)
   Depth 5: Black takes back White Queen immediately after. Eval is actually Equal.

   Solution:
   When Depth is 0, do NOT stop if there are "noisy" moves (usually limited to captures), though some people include checks and promotions.
   Keep searching strictly through noisy moves until the position is "Quiet".
*/

func (p *Position) quiesce(alpha, beta, ply int, tc *TimeControl) int {
	p.localNodes++

	if (p.localNodes & NodeCheckMaskSearch) == 0 {
		if tc.shouldStop() {
			return alpha
		}
	}

	inCheck := p.inCheck()
	best := alpha
	if !inCheck {
		stand := p.evaluate()
		if stand >= beta {
			return stand
		}
		best = stand
		if stand > alpha {
			alpha = stand
		}
		if !p.isEndgame() {
			them := p.side ^ 1
			maxGain := 0
			if p.pieces[them][Queen] != 0 {
				maxGain = pieceValues[Queen]
			} else if p.pieces[them][Rook] != 0 {
				maxGain = pieceValues[Rook]
			} else if (p.pieces[them][Bishop] | p.pieces[them][Knight]) != 0 {
				maxGain = pieceValues[Bishop]
			} else if p.pieces[them][Pawn] != 0 {
				maxGain = pieceValues[Pawn]
			}
			if stand+maxGain+DeltaMargin < alpha {
				return stand
			}
		}
	}

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], !inCheck)
	moves := movesArr[:n]
	moves = p.orderMoves(moves, 0, 0, 0)

	legalCount := 0
	for _, m := range moves {
		if tc.shouldStop() {
			return alpha
		}
		if !p.isLegal(m) {
			continue
		}
		legalCount++

		undo := p.makeMove(m)
		score := -p.quiesce(-beta, -alpha, ply+1, tc)
		p.unmakeMove(m, undo)

		if score >= beta {
			return beta
		}
		if score > best {
			best = score
		}
		if score > alpha {
			alpha = score
		}
	}

	if inCheck && legalCount == 0 {
		return -Mate + ply
	}

	return best
}

/*
  ----------------------------------------------------------------------------------
   NEGAMAX SEARCH WITH ALPHA-BETA PRUNING
  ----------------------------------------------------------------------------------
   This function explores the game tree to find the best move. It uses the Negamax
   framework where max(a, b) = -min(-b, -a), simplifying code for 2-player zero-sum games.

          [ Root Node ]
          /     |     \
      [Move A] [Move B] [Move C]
        /         |        \
     ...         ...      (Pruned?)

   Key Optimizations used here:
   1. Transposition Table (TT): Cache results to avoid re-searching identical positions.
   2. Null Move Pruning (NMP): "Pass" the move; if still safe, the position is too good (cutoff).
   3. Late Move Reductions (LMR): Search moves that were ordered as less promising at lower depth first.
   4. Quiescence Search: At leaf nodes, play out captures to avoid "horizon effects".

   Alpha (α): Best score the maximizing player can guarantee so far.
   Beta  (β): Best score the minimizing player can guarantee so far.
   Condition: If Score >= Beta, we have a "Cutoff" (branch is too good, opponent won't allow it).
*/

func (p *Position) negamax(depth, alpha, beta, ply int, pv *[]Move, tc *TimeControl, ss *[MaxDepth + 100]SearchStack) int {

	p.localNodes++

	// Time check
	if (p.localNodes & NodeCheckMaskSearch) == 0 {
		if tc.shouldStop() {
			return alpha
		}
	}

	inCheck := p.inCheck()
	if inCheck {
		depth++
	}

	if depth <= 0 {
		return p.quiesce(alpha, beta, ply, tc)
	}

	// Transposition table lookup
	origAlpha := alpha
	var hashMove Move
	if e, found, usable := tt.Probe(p.hash, depth); found {
		move, score, _, _, flag := e.unpack()
		hashMove = Move(move)

		if pv == nil {
			scoreFromTT := int(score)
			isMateScore := scoreFromTT > Mate-MateScoreGuard || scoreFromTT < -Mate+MateScoreGuard
			if isMateScore {
				if scoreFromTT > 0 {
					scoreFromTT -= ply
				} else {
					scoreFromTT += ply
				}
			}

			if usable {
				switch flag {
				case ttFlagExact:
					return scoreFromTT
				case ttFlagLower:
					if scoreFromTT >= beta {
						return scoreFromTT
					}
				case ttFlagUpper:
					if scoreFromTT <= alpha {
						return scoreFromTT
					}
				}
			}
		}
	}

	// IIR
	if depth >= 4 && hashMove == 0 {
		depth--
	}

	// Mate distance pruning
	if beta > Mate-ply {
		if alpha >= Mate-ply {
			return Mate - ply
		}
		beta = Mate - ply
	}
	if alpha < -Mate+ply {
		if -Mate+ply >= beta {
			return -Mate + ply
		}
		alpha = -Mate + ply
	}

	// RFP check
	if depth <= 8 && !inCheck && !p.isEndgame() {
		if hashMove != 0 && !hashMove.isCapture() {
			eval := p.evaluate()
			// If we are far above beta, we can return soft fail
			if eval >= beta+DeltaMargin*depth {
				return eval
			}
		}
	}

	// Null move pruning
	if depth >= 3 && !inCheck && !p.isEndgame() && pv == nil {
		R := 2
		undo := p.makeNullMove()
		score := -p.negamax(depth-1-R, -beta, -beta+1, ply+1, nil, tc, ss)
		p.unmakeNullMove(undo)

		if score >= beta {
			return beta
		}
	}

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	moves := p.orderMoves(movesArr[:n], hashMove, ss[ply].killer1, ss[ply].killer2)

	bestMove := Move(0)
	bestScore := -Infinity
	legalMoves := 0

	pvNode := pv != nil
	var pvPtr *[]Move

	var childPVBuf [MaxDepth]Move

	for _, m := range moves {
		if tc.shouldStop() {
			return alpha
		}
		if !p.isLegal(m) {
			continue
		}
		legalMoves++
		undo := p.makeMove(m)
		childPV := childPVBuf[:0]
		if pvNode {
			pvPtr = &childPV
		}
		var score int
		if p.isDraw() {
			score = 0
		} else {
			// Late move reductions & Principal variation search
			childDepth := depth - 1
			k := &ss[ply]
			isKiller := m == k.killer1 || m == k.killer2
			canReduce := childDepth >= LMRMinChildDepth && !inCheck && !m.isCapture() && !m.isPromo() && legalMoves > LMRLateMoveAfter && !isKiller
			var eff int
			if canReduce {
				red := 1 + (childDepth-LMRMinChildDepth)/6 + (legalMoves-3)/6
				eff = childDepth - red
			}
			if canReduce && eff >= 1 {
				score = -p.negamax(eff, -alpha-1, -alpha, ply+1, nil, tc, ss)
				if score > alpha {
					score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss)
				}
			} else {
				if legalMoves > 1 && pvNode {
					score = -p.negamax(childDepth, -alpha-1, -alpha, ply+1, nil, tc, ss)
					if score > alpha {
						score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss)
					}
				} else {
					score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss)
				}
			}
		}

		p.unmakeMove(m, undo)

		if score >= beta {
			// Update killers
			if !m.isCapture() && !m.isPromo() && m != hashMove {
				k := &ss[ply]
				if m != k.killer1 {
					k.killer2, k.killer1 = k.killer1, m
				}
			}

			// Store in transposition table
			storeScore := score
			if storeScore > Mate-MateScoreGuard {
				storeScore += ply
			} else if storeScore < -Mate+MateScoreGuard {
				storeScore -= ply
			}
			tt.Save(p.hash, m, storeScore, depth, ttFlagLower)
			return beta
		}

		// Update best move
		if score > bestScore {
			bestScore = score
			bestMove = m
		}

		// Update alpha
		if score > alpha {
			alpha = score
			if pv != nil {
				*pv = append(append((*pv)[:0], m), childPV...)
			}
		}
	}

	// Handle draw & checkmate results
	if legalMoves == 0 {
		var result int
		if inCheck {
			result = -Mate + ply
		} else {
			result = 0
		}
		storeScore := result
		if storeScore > Mate-MateScoreGuard {
			storeScore += ply
		} else if storeScore < -Mate+MateScoreGuard {
			storeScore -= ply
		}
		tt.Save(p.hash, Move(0), storeScore, depth, ttFlagExact)
		return result
	}

	flag := ttFlagExact
	if bestScore <= origAlpha {
		flag = ttFlagUpper
	}

	storeScore := bestScore
	if storeScore > Mate-MateScoreGuard {
		storeScore += ply
	} else if storeScore < -Mate+MateScoreGuard {
		storeScore -= ply
	}
	tt.Save(p.hash, bestMove, storeScore, depth, flag)
	return bestScore
}

/*
  ----------------------------------------------------------------------------------
   ITERATIVE DEEPENING SEARCH
  ----------------------------------------------------------------------------------
   Instead of searching directly to Depth 10, we search Depth 1, then 2, then 3...
   This might seem slower than just going directly to depth 10, but it offers unique advantages:

   Visual Timeline:
   [Start] ->
    Search D=1 -> BestMove A
	Search D=2 -> BestMove A (uses info from depth 1 to sort moves)
	Search D=3 -> BestMove B (found a better move)
	[Time Up!] -> Return result, even if iteration is unfinished. This is safe because even though its an unfinished iteration,
	We searched the previous completed iteration fully, sorted that move as likely the best for this iteration as well,
	so we either return that or an even better move.

   Benefits:
   1. Time Management: We always have a "best move so far" if we must stop abruptly (as we do in chess).
   2. Move Ordering: The BestMove from Depth X-1 is the first move searched at Depth X.
*/

func (p *Position) search(tc *TimeControl) Move {
	var bestMove Move
	var ss [MaxDepth + 100]SearchStack

	maxDepth := tc.depth
	if maxDepth == 0 || tc.infinite {
		maxDepth = MaxDepth
	}

	// Start timers
	p.localNodes = 0
	start := time.Now()
	var prevScore int
	var pvBuf [MaxDepth]Move
	for depth := 1; depth <= maxDepth; depth++ {
		pv := pvBuf[:0]
		var score int

		// Aspiration windows
		needFull := false
		if depth >= AspirationStartDepth {
			base := prevScore
			window := AspirationBase
			low := base - window
			high := base + window
			score = p.negamax(depth, low, high, 0, &pv, tc, &ss)
			if score <= low || score >= high {
				needFull = true
			}
		} else {
			needFull = true
		}
		if needFull {
			pv = pvBuf[:0]
			score = p.negamax(depth, -Infinity, Infinity, 0, &pv, tc, &ss)
		}
		prevScore = score
		iterNodes := p.localNodes

		elapsed := time.Since(start)
		elapsedMs := elapsed.Milliseconds()
		nps := int64(0)
		if elapsed > 0 {
			nps = int64(float64(iterNodes) / elapsed.Seconds())
		}

		if tc.shouldStop() {
			break
		}

		if len(pv) > 0 {
			bestMove = pv[0]
		}

		// Print search info
		absScore := abs(score)
		if absScore >= Mate-MateScoreGuard {
			matePly := Mate - absScore
			mateMoves := (matePly + 1) / 2
			if score > 0 {
				fmt.Printf("info depth %d score mate %d nodes %d time %d nps %d pv",
					depth, mateMoves, iterNodes, elapsedMs, nps)
			} else {
				fmt.Printf("info depth %d score mate -%d nodes %d time %d nps %d pv",
					depth, mateMoves, iterNodes, elapsedMs, nps)
			}
		} else {
			fmt.Printf("info depth %d score cp %d nodes %d time %d nps %d pv",
				depth, score, iterNodes, elapsedMs, nps)
		}
		for _, m := range pv {
			fmt.Printf(" %v", m)
		}
		fmt.Println()

		// Return result
		if absScore >= Mate-MateScoreGuard {
			return bestMove
		}

		if !tc.shouldContinue(elapsed) {
			break
		}
	}
	return bestMove
}

// -------------------------------------------
// TIME CONTROL
// -------------------------------------------
func (tc *TimeControl) Stop() {
	atomic.StoreInt32(&tc.stopped, 1)
}

/*
  ----------------------------------------------------------------------------------
   TIME MANAGEMENT HEURISTICS
  ----------------------------------------------------------------------------------
   Deciding how much time to spend on a move is hard balance to find.
   - Too little: We play hasty, weak moves.
   - Too much: We likely flag (run out of time) later in the game.

   Strategy:
   1. Moves To Go: Assume the game lasts ~20-30-40 more moves (Can be whatever you like).
   2. Increment: Always safely bank on the increment (winc/binc).
   3. Safety Buffer: Subtract a margin (minTimeMs) to account for possible GUI lag.

   Formula:
    TimeForMove = (RemainingTime / MovesToGo) + Increment
*/

func (tc *TimeControl) allocateTime(side int) {
	if tc.movetime > 0 {
		tc.deadline = time.Now().Add(time.Duration(tc.movetime) * time.Millisecond)
		return
	}
	if tc.infinite || tc.depth > 0 {
		tc.deadline = time.Time{}
		return
	}

	var myTime, myInc int64
	if side == White {
		myTime, myInc = tc.wtime, tc.winc
	} else {
		myTime, myInc = tc.btime, tc.binc
	}

	movesToGo := tc.movestogo
	if movesToGo <= 0 {
		movesToGo = DefaultMovesToGo
	}

	usableTime := max(myTime-minTimeMs, 0)

	fromBank := usableTime / int64(movesToGo)
	capBank := usableTime / perMoveCapDiv
	if fromBank > capBank {
		fromBank = capBank
	}

	baseMs := min(fromBank+myInc, usableTime)
	if baseMs < minTimeMs && usableTime > 0 {
		baseMs = min(minTimeMs, usableTime)
	}

	tc.deadline = time.Now().Add(time.Millisecond * time.Duration(baseMs))
}

func (tc *TimeControl) shouldStop() bool {
	if atomic.LoadInt32(&tc.stopped) != 0 {
		return true
	}
	d := tc.deadline
	return !d.IsZero() && time.Until(d) <= 0
}

func (tc *TimeControl) shouldContinue(lastIter time.Duration) bool {
	if atomic.LoadInt32(&tc.stopped) != 0 {
		return false
	}

	if tc.infinite || tc.depth > 0 || lastIter <= 0 {
		return true
	}

	remain := time.Until(tc.deadline)
	if remain <= 0 {
		return false
	}

	minRequired := lastIter*nextIterMult + continueMargin
	return remain > minRequired
}

/*
  ----------------------------------------------------------------------------------
   PERFT (Performance Test & Move Generation Validator)
  ----------------------------------------------------------------------------------
   Perft is a debugging function that traverses the move tree to a specific depth
   and counts the number of leaf nodes.

   Why is this useful?
   It verifies that the move generator (generateMovesTo, makeMove, unmakeMove) is
   mostly bug-free. We compare the results against known values for the start position.

   Example (Depth 1):
   Start Pos -> 20 legal moves. Perft(1) should return 20.

   Divide:
   "Divide" prints the child count for *each* root move separately.
   Perft 2 Divide Example:
   e2e4: 20
   e2e3: 20
   g1f3: 20
   --------
   Total: 400
   This isolates exactly which move branch contains a possible bug, if node counts differ from known results.
*/

func (p *Position) perft(depth int) int {
	if depth == 0 {
		return 1
	}

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	count := 0
	for i := 0; i < n; i++ {
		m := movesArr[i]
		if !p.isLegal(m) {
			continue
		}
		undo := p.makeMove(m)
		count += p.perft(depth - 1)
		p.unmakeMove(m, undo)
	}

	return count
}

func (p *Position) perftDivide(depth int) {
	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	total := 0

	for i := 0; i < n; i++ {
		m := movesArr[i]
		if !p.isLegal(m) {
			continue
		}
		undo := p.makeMove(m)
		count := p.perft(depth - 1)
		p.unmakeMove(m, undo)

		fmt.Printf("%v: %d\n", m, count)
		total += count
	}

	fmt.Printf("\nTotal: %d\n", total)
}

func runSearchAndReport(p *Position, tc *TimeControl) {
	move := p.search(tc)
	if !currentTC.CompareAndSwap(tc, nil) {
		return
	}
	fmt.Println("bestmove", move)
}

func parseSetOption(parts []string) (name, value string) {
	nameStart, nameEnd, valueStart := -1, -1, -1
	for i, p := range parts {
		if p == "name" && nameStart == -1 {
			nameStart = i + 1
			continue
		}
		if p == "value" && nameStart != -1 && nameEnd == -1 {
			nameEnd = i
			valueStart = i + 1
			break
		}
	}
	if nameStart == -1 {
		return "", ""
	}
	if nameEnd == -1 {
		return strings.Join(parts[nameStart:], " "), ""
	}

	if nameStart >= nameEnd {
		return "", ""
	}
	name = strings.Join(parts[nameStart:nameEnd], " ")
	value = strings.Join(parts[valueStart:], " ")
	return name, value
}

/*
  ----------------------------------------------------------------------------------
   UCI MAIN LOOP (Universal Chess Interface)
  ----------------------------------------------------------------------------------
   This is the communication part. The GUI (Arena, Banksia, Cutechess) sends text commands, we reply with text.

   [ GUI ] -------- "position startpos moves e2e4" ---->  [ ENGINE ]
   [ GUI ] <------- "info depth 5 score cp 20..." ------  [ ENGINE ]
   [ GUI ] -------- "go wtime 60000" ------------------>  [ ENGINE ]
   [ GUI ] <------- "bestmove e7e5" --------------------  [ ENGINE ]

   The loop waits for Stdin input, parses the string, and triggers engine functions.
   It must be non-blocking where possible to handle "stop" commands.
*/

func uciLoop() {
	pos := NewPosition()
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Fprintln(os.Stderr, "# Soomi V1.1.9 ready. Type 'help' for available commands.")

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		cmd := parts[0]

		switch cmd {
		case "uci":
			fmt.Println("id name Soomi V1.1.9")
			fmt.Println("id author Otto Laukkanen")
			fmt.Println("option name Hash type spin default 256 min 1 max 4096")
			fmt.Println("uciok")

		case "isready":
			fmt.Println("readyok")

		case "setoption":
			name, value := parseSetOption(parts)
			if strings.EqualFold(name, "Hash") {
				sizeMB, err := strconv.Atoi(value)
				if err != nil || sizeMB <= 0 {
					fmt.Printf("info string invalid hash value: %s\n", value)
					continue
				}
				if cur := currentTC.Load(); cur != nil {
					cur.Stop()
					if currentTC.Load() != nil {
						fmt.Printf("info string Hash unchanged (search running)\n")
						continue
					}
				}
				InitTT(sizeMB)
				fmt.Printf("info string Hash set to %d MB\n", sizeMB)
			} else {
				fmt.Printf("info string setoption %q = %q (ignored)\n", name, value)
			}

		case "ucinewgame":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			tt.Clear()
			pos.setStartPos()

		case "position":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			if len(parts) < 2 {
				fmt.Println("# Error: position requires arguments")
				continue
			}
			moveIdx := -1
			for i := 2; i < len(parts); i++ {
				if parts[i] == "moves" {
					moveIdx = i
					break
				}
			}

			pos.setStartPos()
			if parts[1] != "startpos" {
				fmt.Println("info string only 'position startpos [moves ...]' is supported; resetting to startpos")
				break
			}

			if moveIdx != -1 && moveIdx+1 < len(parts) {
				for _, mvStr := range parts[moveIdx+1:] {
					if len(mvStr) < 4 || len(mvStr) > 5 {
						fmt.Printf("# Error: invalid move format: %s. Further moves ignored.\n", mvStr)
						break
					}
					found := false
					var buf [256]Move
					n := pos.generateMovesTo(buf[:], false)
					for j := 0; j < n; j++ {
						m := buf[j]
						if strings.EqualFold(m.String(), mvStr) && pos.isLegal(m) {
							pos.makeMove(m)
							found = true
							break
						}
					}
					if !found {
						fmt.Printf("# Error: illegal move: %s. Further moves ignored.\n", mvStr)
						break
					}
				}
			}
		case "go":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}

			tc := &TimeControl{}

			for i := 1; i < len(parts); i++ {
				switch parts[i] {
				case "wtime":
					if i+1 < len(parts) {
						tc.wtime, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "btime":
					if i+1 < len(parts) {
						tc.btime, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "winc":
					if i+1 < len(parts) {
						tc.winc, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "binc":
					if i+1 < len(parts) {
						tc.binc, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "movestogo":
					if i+1 < len(parts) {
						tc.movestogo, _ = strconv.Atoi(parts[i+1])
						i++
					}
				case "depth":
					if i+1 < len(parts) {
						tc.depth, _ = strconv.Atoi(parts[i+1])
						i++
					}
				case "movetime":
					if i+1 < len(parts) {
						tc.movetime, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "infinite":
					tc.infinite = true
				}
			}

			tc.allocateTime(pos.side)
			pcopy := *pos
			currentTC.Store(tc)

			go runSearchAndReport(&pcopy, tc)

		case "stop":
			if cur := currentTC.Load(); cur != nil {
				cur.Stop()
			}

		case "quit":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			return

		case "d", "display":
			fmt.Println("\n   a b c d e f g h")
			fmt.Println("  ----------------")
			for r := 7; r >= 0; r-- {
				fmt.Printf("%d|", r+1)
				for f := 0; f < 8; f++ {
					sq := r*8 + f
					c, pt, ok := pos.pieceAt(sq)
					if ok {
						piece := "PNBRQK"[pt]
						if c == Black {
							piece += 32
						}
						fmt.Printf(" %c", piece)
					} else {
						fmt.Print(" .")
					}
				}
				fmt.Printf(" |%d\n", r+1)
			}
			fmt.Println("  ----------------")
			fmt.Println("   a b c d e f g h")
			fmt.Printf("Side to move: %s\n", map[int]string{White: "White", Black: "Black"}[pos.side])
			fmt.Printf("Hash: %x\n\n", pos.hash)

		case "eval":
			score := pos.evaluate()
			fmt.Printf("Evaluation: %+d (from %s's perspective)\n", score, map[int]string{White: "White", Black: "Black"}[pos.side])

		case "perft":
			if len(parts) > 1 {
				maxDepth, _ := strconv.Atoi(parts[1])
				fmt.Println("\nRunning perft test...")
				fmt.Println("Depth    Nodes           Time        NPS")
				fmt.Println("---------------------------------------------")
				for depth := 1; depth <= maxDepth; depth++ {
					start := time.Now()
					count := pos.perft(depth)
					elapsed := time.Since(start)
					nps := int64(0)
					if elapsed.Seconds() > 0 {
						nps = int64(float64(count) / elapsed.Seconds())
					}
					timeStr := ""
					if elapsed < time.Second {
						timeStr = fmt.Sprintf("%d ms", elapsed.Milliseconds())
					} else {
						timeStr = fmt.Sprintf("%.2f s", elapsed.Seconds())
					}
					fmt.Printf("%-8d %-15d %-11s %d\n", depth, count, timeStr, nps)
				}
				fmt.Println()
			} else {
				fmt.Println("# Usage: perft <depth>")
			}

		case "divide":
			if len(parts) > 1 {
				depth, _ := strconv.Atoi(parts[1])
				pos.perftDivide(depth)
			} else {
				fmt.Println("# Usage: divide <depth>")
			}

		case "help":
			printHelp()

		default:
			fmt.Printf("# Unknown command: %s (type 'help' for available commands)\n", cmd)
		}
	}
}

func printHelp() {
	fmt.Println(`# Soomi V1.1.9 - Available Commands:

UCI Protocol Commands:
  uci                              - Initialize UCI mode
  isready                          - Check if engine is ready
  ucinewgame                       - Start a new game
  position startpos                - Set starting position
  position startpos moves <moves>  - Set position after moves
  go [options]                     - Start searching
      wtime <ms>                   - White's remaining time
      btime <ms>                   - Black's remaining time
      winc <ms>                    - White's increment per move
      binc <ms>                    - Black's increment per move
      movestogo <n>                - Moves until time control
      depth <n>                    - Search to fixed depth
      movetime <ms>                - Search for fixed time
      infinite                     - Search indefinitely
  stop                             - Stop searching
  quit                             - Exit engine

Additional Commands:
  d, display                       - Display current board
  eval                             - Show static evaluation
  perft <depth>                    - Run perft test
  divide <depth>                   - Run divide test
  help                             - Show this help message

Example Usage:
  1. Start new game:
     ucinewgame
     
  2. Set position and make moves:
     position startpos moves e2e4 e7e5 g1f3
     
  3. Search with time control:
     go wtime 300000 btime 300000 winc 0 binc 0
     
  4. Search to depth 10:
     go depth 10
     
  5. Display current position:
     d`)
}

func main() {
	fmt.Fprintln(os.Stderr, "Soomi V1.1.9 - UCI Chess Engine")
	fmt.Fprintln(os.Stderr, "Type 'help' for available commands or 'uci' to enter UCI mode")
	fmt.Fprintln(os.Stderr)
	uciLoop()
}

// To make an executable
// go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi-V1.1.9.exe soomi.go
