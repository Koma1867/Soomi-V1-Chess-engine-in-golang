package main

import (
	"bufio"
	"fmt"
	"math/bits"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	White = 0
	Black = 1
)

const (
	Pawn   = 0
	Knight = 1
	Bishop = 2
	Rook   = 3
	Queen  = 4
	King   = 5
)

const (
	MaxDepth             = 32
	Infinity             = 30000
	MateValue            = 29000
	Mate                 = MateValue
	AspirationBase       = 30
	AspirationStep       = 3
	AspirationStartDepth = 5
	DefaultMovesToGo     = 30
	NodeCheckMaskSearch  = 2047
	Razor2               = 285
	Razor1               = 201
	DeltaMargin          = 200
	maxLMRMoves          = 32
	LMRMinChildDepth     = 3
	LMRLateMoveAfter     = 2
	MateScoreGuard       = 1000
	MateLikeThreshold    = Mate - MateScoreGuard
)

const (
	minTimeMs      int64         = 10
	perMoveCapDiv  int64         = 3
	nextIterMult                 = 3
	continueMargin time.Duration = 10 * time.Millisecond
)

const (
	FlagQuiet   = 0
	FlagCapture = 4
	FlagEP      = 5
	FlagCastle  = 2
	FlagPromoN  = 8
	FlagPromoB  = 9
	FlagPromoR  = 10
	FlagPromoQ  = 11
	FlagPromoCN = 12
	FlagPromoCB = 13
	FlagPromoCR = 14
	FlagPromoCQ = 15
)

const (
	scoreHash            = 1000000
	scorePromoBase       = 900000
	scoreCaptureBase     = 800000
	scoreKiller1         = 750000
	scoreKiller2         = 740000
	scoreFallbackCapture = 700000
)

const (
	ttFlagExact uint8 = 0
	ttFlagLower uint8 = 1
	ttFlagUpper uint8 = 2
)

const defaultTTSizeMB = 256

const MaxGamePly = 1024

const (
	KnightMobZeroPoint = 4
	KnightMobCpPerMove = 3
	BishopMobZeroPoint = 7
	BishopMobCpPerMove = 2
	RookMobZeroPoint   = 7
	RookMobCpPerMove   = 2
	QueenMobZeroPoint  = 14
	QueenMobCpPerMove  = 1
	EgMobCpPerMove     = 2
	SafetyTableSize    = 100
	KnightAttackWeight = 2
	BishopAttackWeight = 2
	RookAttackWeight   = 3
	QueenAttackWeight  = 5
	PhaseScale         = 256
	MVVLVAWeight       = 100
	MarginLMP          = 75
	LMPMoveThreshold   = 8
	LMPMaxDepth        = 3
)

var piecePhase = [6]int{0, 1, 1, 2, 4, 0}
var totalPhase = piecePhase[Pawn]*16 + piecePhase[Knight]*4 + piecePhase[Bishop]*4 + piecePhase[Rook]*4 + piecePhase[Queen]*2

var (
	pieceValues = [6]int{100, 320, 330, 500, 950, 20000}
	pst         [2][6][64]int
	pstEnd      [2][6][64]int
)

var (
	PassedPawnMG = [8]int{0, 5, 12, 22, 36, 56, 84, 0}
	PassedPawnEG = [8]int{0, 8, 20, 36, 62, 98, 154, 0}
)

const ZobristSeed = 1070372

var (
	currentTC atomic.Pointer[TimeControl]
	tt        *TranspositionTable
)

type MagicEntry struct {
	mask   Bitboard
	magic  Bitboard
	shift  uint8
	offset uint32
}

var (
	rookMagics        [64]MagicEntry
	bishopMagics      [64]MagicEntry
	rookAttackTable   [102400]Bitboard
	bishopAttackTable [5248]Bitboard
)

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
	fullmove         int
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
	hash             uint64
	castle           int
	epSquare         int
	halfmove         int
	captured         int
	lastIrreversible int
	historyPly       int
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

var (
	safetyTable  [SafetyTableSize]int
	kingZoneMask [2][64]Bitboard
)

var (
	knightMobilityMG [9]int
	bishopMobilityMG [14]int
	rookMobilityMG   [15]int
	queenMobilityMG  [28]int
	mobilityEG       [28]int
)

var (
	passedPawnBonusMG [8]int
	passedPawnBonusEG [8]int
	fileMask          [8]Bitboard
)

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
	if ph > totalPhase {
		return totalPhase
	}
	return ph
}

func (p *Position) isEndgame() bool {
	phase := p.computePhase()
	return phase < totalPhase/4
}

func phaseScale(ph int) int {
	return ((totalPhase-ph)*PhaseScale + totalPhase/2) / totalPhase
}

func initSafetyTable() {
	for i := 0; i < SafetyTableSize; i++ {
		safetyTable[i] = (i * i) / 4
		if safetyTable[i] > 150 {
			safetyTable[i] = 150
		}
	}
}

func initKingZoneMasks() {
	for sq := 0; sq < 64; sq++ {
		baseZone := kingAttacks[sq] | sqBB[sq]

		r := sq / 8

		whiteZone := baseZone
		if r < 6 {
			whiteZone |= sqBB[sq+8]
			if r < 5 {
				whiteZone |= sqBB[sq+16]
			}
			if r < 4 {
				whiteZone |= sqBB[sq+24]
			}
		}
		kingZoneMask[White][sq] = whiteZone

		blackZone := baseZone
		if r > 1 {
			blackZone |= sqBB[sq-8]
			if r > 2 {
				blackZone |= sqBB[sq-16]
			}
			if r > 3 {
				blackZone |= sqBB[sq-24]
			}
		}
		kingZoneMask[Black][sq] = blackZone
	}
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

var lmrTable [][]int

func initLMR() {
	maxD := MaxDepth
	rows := maxD + 1
	rowLen := maxLMRMoves + 1
	flat := make([]int, rows*rowLen)
	lmrTable = make([][]int, rows)
	for d := 0; d <= maxD; d++ {
		offset := d * rowLen
		lmrTable[d] = flat[offset : offset+rowLen]
		if d >= LMRMinChildDepth {
			depthBonus := (d - LMRMinChildDepth) / 10
			for m := 3; m <= maxLMRMoves; m++ {
				lmrTable[d][m] = 1 + depthBonus + (m-3)/6
			}
		}
	}
}

func initMobility() {
	for i := range knightMobilityMG {
		knightMobilityMG[i] = (i - KnightMobZeroPoint) * KnightMobCpPerMove
	}
	for i := range bishopMobilityMG {
		bishopMobilityMG[i] = (i - BishopMobZeroPoint) * BishopMobCpPerMove
	}
	for i := range rookMobilityMG {
		rookMobilityMG[i] = (i - RookMobZeroPoint) * RookMobCpPerMove
	}
	for i := range queenMobilityMG {
		queenMobilityMG[i] = (i - QueenMobZeroPoint) * QueenMobCpPerMove
	}
	for i := range mobilityEG {
		mobilityEG[i] = i * EgMobCpPerMove
	}
}

func initPassedPawns() {
	for i := 0; i < 8; i++ {
		fileMask[i] = 0x0101010101010101 << i
	}

	passedPawnBonusMG = PassedPawnMG
	passedPawnBonusEG = PassedPawnEG
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
	mvvLvaFlat      [36]int
)

func (p *Position) isRepetition() bool {
	if p.historyPly-p.lastIrreversible < 2 {
		return false
	}
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
	if entries < 1 {
		entries = 1
	}
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
	depth = max(0, min(depth, 63))
	score = max(min(score, 32767), -32768)
	newPacked := packEntry(uint32(mv), int16(score), uint8(t.gen), uint8(depth), flag)
	idx := int(key & t.mask)
	old := t.entries[idx]
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
	initMVVLVATable()
	initLMR()
	initMobility()
	initPassedPawns()
	initSafetyTable()
	initKingZoneMasks()
	InitTT(defaultTTSizeMB)
}

func initPST() {
	pst[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		-6, 6, 6, -14, -14, 6, 6, -6,
		-6, 0, -8, 6, 6, -8, 0, -6,
		0, 2, 10, 22, 22, 10, 2, 0,
		6, 6, 16, 26, 26, 16, 6, 6,
		10, 12, 20, 30, 30, 20, 12, 10,
		20, 20, 20, 20, 20, 20, 20, 20,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pst[White][Knight] = [64]int{
		-50, -40, -30, -30, -30, -30, -40, -50,
		-40, -20, 0, 6, 6, 0, -20, -40,
		-30, 6, 12, 16, 16, 12, 6, -30,
		-30, 0, 16, 22, 22, 16, 0, -30,
		-30, 6, 16, 24, 24, 16, 6, -30,
		-30, 0, 12, 16, 16, 12, 0, -30,
		-40, -20, 0, 0, 0, 0, -20, -40,
		-50, -40, -30, -30, -30, -30, -40, -50,
	}

	pst[White][Bishop] = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 18, 0, 0, 0, 0, 18, -10,
		-10, 0, 6, 12, 12, 6, 0, -10,
		-10, 6, 6, 12, 12, 6, 6, -10,
		-10, 0, 12, 16, 16, 12, 0, -10,
		-10, 10, 10, 16, 16, 10, 10, -10,
		-10, 6, 0, 0, 0, 0, 6, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}

	pst[White][Rook] = [64]int{
		0, 0, 0, 5, 5, 0, 0, 0,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		10, 12, 12, 15, 15, 12, 12, 10,
		0, 0, 0, 5, 5, 0, 0, 0,
	}

	pst[White][Queen] = [64]int{
		-20, -10, -10, -5, -5, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 4, 4, 4, 4, 0, -10,
		-5, 0, 4, 4, 4, 4, 0, -5,
		0, 0, 4, 4, 4, 4, 0, -5,
		-10, 4, 4, 4, 4, 4, 0, -10,
		-10, 0, 4, 0, 0, 0, 0, -10,
		-20, -10, -10, -5, -5, -10, -10, -20,
	}

	pst[White][King] = [64]int{
		50, 40, 5, -20, -20, 5, 40, 50,
		10, 10, -15, -30, -30, -15, 10, 10,
		-10, -20, -20, -20, -20, -20, -20, -10,
		-20, -30, -30, -40, -40, -30, -30, -20,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
	}

	copy(pstEnd[:], pst[:])
	pstEnd[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		4, 4, 4, 4, 4, 4, 4, 4,
		10, 10, 10, 10, 10, 10, 10, 10,
		20, 20, 20, 20, 20, 20, 20, 20,
		34, 34, 34, 34, 34, 34, 34, 34,
		48, 48, 48, 48, 48, 48, 48, 48,
		60, 60, 60, 60, 60, 60, 60, 60,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pstEnd[White][Knight] = [64]int{
		-48, -38, -28, -28, -28, -28, -38, -48,
		-38, -18, 0, 2, 2, 0, -18, -38,
		-28, 2, 10, 14, 14, 10, 2, -28,
		-28, 6, 14, 20, 20, 14, 6, -28,
		-28, 2, 14, 20, 20, 14, 2, -28,
		-28, 6, 10, 14, 14, 10, 6, -28,
		-38, -18, 0, 4, 4, 0, -18, -38,
		-48, -38, -28, -28, -28, -28, -38, -48,
	}

	pstEnd[White][Bishop] = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 6, 12, 12, 6, 0, -10,
		-10, 6, 12, 16, 16, 12, 6, -10,
		-10, 0, 12, 16, 16, 12, 0, -10,
		-10, 6, 12, 16, 16, 12, 6, -10,
		-10, 10, 6, 6, 6, 6, 10, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}

	pstEnd[White][Rook] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		6, 10, 10, 10, 10, 10, 10, 6,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		15, 20, 20, 25, 25, 20, 20, 15,
		10, 10, 10, 10, 10, 10, 10, 10,
	}

	pstEnd[White][Queen] = [64]int{
		-20, -10, -10, -5, -5, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 4, 4, 4, 4, 0, -10,
		-5, 0, 4, 4, 4, 4, 0, -5,
		-5, 0, 4, 4, 4, 4, 0, -5,
		-10, 0, 4, 4, 4, 4, 0, -10,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-20, -10, -10, -5, -5, -10, -10, -20,
	}

	pstEnd[White][King] = [64]int{
		-50, -40, -30, -20, -20, -30, -40, -50,
		-30, -20, -10, 0, 0, -10, -20, -30,
		-30, -10, 20, 30, 30, 20, -10, -30,
		-30, -10, 30, 40, 40, 30, -10, -30,
		-30, -10, 30, 40, 40, 30, -10, -30,
		-30, -10, 20, 30, 30, 20, -10, -30,
		-30, -30, 0, 0, 0, 0, -30, -30,
		-50, -40, -30, -20, -20, -30, -40, -50,
	}

	for pt := 0; pt < 6; pt++ {
		for sq := 0; sq < 64; sq++ {
			bsq := sq ^ 56
			pst[Black][pt][sq] = pst[White][pt][bsq]
			pstEnd[Black][pt][sq] = pstEnd[White][pt][bsq]
		}
	}
}

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

func initMVVLVATable() {
	for a := 0; a < 5; a++ {
		for v := 0; v < 5; v++ {
			mvvLvaFlat[a*6+v] = pieceValues[v]*MVVLVAWeight - pieceValues[a]
		}
	}
}

func mvvLvaScore(att, vic int) int {
	if att == King || vic == King {
		return 0
	}
	return int(mvvLvaFlat[att*6+vic])
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

func nullMoveReduction(depth int) int {
	return 3 + min(2, depth/6)
}

func makeMove(from, to, flags int) Move {
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
	p.fullmove = 1

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

	if p.castle&1 != 0 {
		p.hash ^= zobristCastleWK
	}
	if p.castle&2 != 0 {
		p.hash ^= zobristCastleWQ
	}
	if p.castle&4 != 0 {
		p.hash ^= zobristCastleBK
	}
	if p.castle&8 != 0 {
		p.hash ^= zobristCastleBQ
	}
	p.historyPly = 0
	p.lastIrreversible = 0
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
	pawns := p.pieces[bySide][Pawn]
	if pawnAttacks[bySide^1][sq]&pawns != 0 {
		return true
	}
	if knightAttacks[sq]&p.pieces[bySide][Knight] != 0 {
		return true
	}
	if kingAttacks[sq]&p.pieces[bySide][King] != 0 {
		return true
	}
	bish := p.pieces[bySide][Bishop]
	qu := p.pieces[bySide][Queen]
	if bishopAttacks(sq, p.all)&(bish|qu) != 0 {
		return true
	}
	rook := p.pieces[bySide][Rook]
	return rookAttacks(sq, p.all)&(rook|qu) != 0
}

func (p *Position) inCheck() bool {
	kingBB := p.pieces[p.side][King]
	kingSq := bits.TrailingZeros64(uint64(kingBB))
	return p.isAttacked(kingSq, p.side^1)
}

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
			if !capturesOnly && to >= 0 && to < 64 && occAll&sqBB[to] == 0 {
				buf[i] = makeMove(from, to, FlagPromoQ)
				i++
				buf[i] = makeMove(from, to, FlagPromoR)
				i++
				buf[i] = makeMove(from, to, FlagPromoB)
				i++
				buf[i] = makeMove(from, to, FlagPromoN)
				i++
			}

			attacks := pawnAttacks[us][from] & occThem
			for att := attacks; att != 0; {
				to := popLSB(&att)
				buf[i] = makeMove(from, to, FlagPromoCQ)
				i++
				buf[i] = makeMove(from, to, FlagPromoCR)
				i++
				buf[i] = makeMove(from, to, FlagPromoCB)
				i++
				buf[i] = makeMove(from, to, FlagPromoCN)
				i++
			}
		} else {
			if !capturesOnly && to >= 0 && to < 64 && occAll&sqBB[to] == 0 {
				buf[i] = makeMove(from, to, FlagQuiet)
				i++

				if from>>3 == dblRank {
					to2 := from + dblPush
					if to2 >= 0 && to2 < 64 && occAll&sqBB[to2] == 0 {
						buf[i] = makeMove(from, to2, FlagQuiet)
						i++
					}
				}
			}

			attacks := pawnAttacks[us][from] & occThem
			for att := attacks; att != 0; {
				to := popLSB(&att)
				buf[i] = makeMove(from, to, FlagCapture)
				i++
			}

			if ep >= 0 && ep < 64 {
				if pawnAttacks[us][from]&sqBB[ep] != 0 {
					buf[i] = makeMove(from, ep, FlagEP)
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
			buf[i] = makeMove(from, to, flag)
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
			buf[i] = makeMove(from, to, flag)
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
			buf[i] = makeMove(from, to, flag)
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
			buf[i] = makeMove(from, to, flag)
			i++
		}
	}

	if p.pieces[us][King] != 0 {
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
			buf[i] = makeMove(kingSq, to, flag)
			i++
		}
	}

	if !capturesOnly && !p.inCheck() {
		if us == White {
			if p.castle&1 != 0 && occAll&0x60 == 0 &&
				!p.isAttacked(5, Black) && !p.isAttacked(6, Black) {
				buf[i] = makeMove(4, 6, FlagCastle)
				i++
			}
			if p.castle&2 != 0 && occAll&0x0E == 0 &&
				!p.isAttacked(3, Black) && !p.isAttacked(2, Black) {
				buf[i] = makeMove(4, 2, FlagCastle)
				i++
			}
		} else {
			if p.castle&4 != 0 && occAll&(0x60<<56) == 0 &&
				!p.isAttacked(61, White) && !p.isAttacked(62, White) {
				buf[i] = makeMove(60, 62, FlagCastle)
				i++
			}
			if p.castle&8 != 0 && occAll&(0x0E<<56) == 0 &&
				!p.isAttacked(59, White) && !p.isAttacked(58, White) {
				buf[i] = makeMove(60, 58, FlagCastle)
				i++
			}
		}
	}
	return i
}

func (p *Position) isLegal(m Move) bool {
	if m == 0 || m.from() == m.to() {
		return false
	}
	from, to := m.from(), m.to()

	us, them := p.side, p.side^1
	val := p.square[from]
	if val < 0 || (val>>3) != us {
		return false
	}
	pt := val & 7
	flags := m.flags()

	fromBB, toBB := sqBB[from], sqBB[to]
	occAll := p.all
	ourOcc := p.occupied[us]
	theirOcc := p.occupied[them]

	if ourOcc&toBB != 0 || (p.pieces[them][King]&toBB) != 0 {
		return false
	}

	if flags == FlagCastle {
		if pt != King {
			return false
		}
		d := to - from
		if d != 2 && d != -2 {
			return false
		}
		if (us == White && from != 4) || (us == Black && from != 60) {
			return false
		}
		if d > 0 {
			if us == White {
				if p.castle&1 == 0 || (occAll&(sqBB[from+1]|sqBB[from+2])) != 0 {
					return false
				}
			} else {
				if p.castle&4 == 0 || (occAll&(sqBB[from+1]|sqBB[from+2])) != 0 {
					return false
				}
			}
			if (p.pieces[us][Rook] & sqBB[from+3]) == 0 {
				return false
			}
			if p.isAttacked(from, them) || p.isAttacked(from+1, them) || p.isAttacked(to, them) {
				return false
			}
		} else {
			if us == White {
				if p.castle&2 == 0 || (occAll&(sqBB[from-1]|sqBB[from-2]|sqBB[from-3])) != 0 {
					return false
				}
			} else {
				if p.castle&8 == 0 || (occAll&(sqBB[from-1]|sqBB[from-2]|sqBB[from-3])) != 0 {
					return false
				}
			}
			if (p.pieces[us][Rook] & sqBB[from-4]) == 0 {
				return false
			}
			if p.isAttacked(from, them) || p.isAttacked(from-1, them) || p.isAttacked(to, them) {
				return false
			}
		}
	} else if flags == FlagEP {
		if pt != Pawn || p.epSquare != to {
			return false
		}
		if occAll&toBB != 0 {
			return false
		}
		capSq := to - 8
		if us == Black {
			capSq = to + 8
		}
		if (p.pieces[them][Pawn] & sqBB[capSq]) == 0 {
			return false
		}
		d := to - from
		if us == White {
			if d != 7 && d != 9 {
				return false
			}
		} else {
			if d != -7 && d != -9 {
				return false
			}
		}
	} else {
		switch pt {
		case Pawn:
			step, startRank := 8, 1
			if us == Black {
				step, startRank = -8, 6
			}
			d := to - from
			if m.isCapture() {
				if (theirOcc & toBB) == 0 {
					return false
				}
				if us == White {
					if d != 7 && d != 9 {
						return false
					}
				} else {
					if d != -7 && d != -9 {
						return false
					}
				}
			} else {
				if occAll&toBB != 0 {
					return false
				}
				if d == step {
				} else if d == 2*step {
					if from/8 != startRank || (occAll&sqBB[from+step]) != 0 {
						return false
					}
				} else {
					return false
				}
			}
			tr := to / 8
			if us == White {
				if tr == 7 && !m.isPromo() {
					return false
				}
				if tr != 7 && m.isPromo() {
					return false
				}
			} else {
				if tr == 0 && !m.isPromo() {
					return false
				}
				if tr != 0 && m.isPromo() {
					return false
				}
			}
			if m.isPromo() {
				t := m.promoType()
				if t < Knight || t > Queen {
					return false
				}
			}
		case Knight:
			if (knightAttacks[from] & toBB) == 0 {
				return false
			}
			if m.isCapture() {
				if (theirOcc & toBB) == 0 {
					return false
				}
			} else if (occAll & toBB) != 0 {
				return false
			}
		case Bishop:
			if (bishopAttacks(from, occAll^fromBB) & toBB) == 0 {
				return false
			}
			if m.isCapture() {
				if (theirOcc & toBB) == 0 {
					return false
				}
			} else if (occAll & toBB) != 0 {
				return false
			}
		case Rook:
			if (rookAttacks(from, occAll^fromBB) & toBB) == 0 {
				return false
			}
			if m.isCapture() {
				if (theirOcc & toBB) == 0 {
					return false
				}
			} else if (occAll & toBB) != 0 {
				return false
			}
		case Queen:
			if ((rookAttacks(from, occAll^fromBB) | bishopAttacks(from, occAll^fromBB)) & toBB) == 0 {
				return false
			}
			if m.isCapture() {
				if (theirOcc & toBB) == 0 {
					return false
				}
			} else if (occAll & toBB) != 0 {
				return false
			}
		case King:
			if (kingAttacks[from] & toBB) == 0 {
				return false
			}
			if m.isCapture() {
				if (theirOcc & toBB) == 0 {
					return false
				}
			} else if (occAll & toBB) != 0 {
				return false
			}
		default:
			return false
		}
	}

	ourP, ourN, ourB, ourR, ourQ, ourK := p.pieces[us][Pawn], p.pieces[us][Knight], p.pieces[us][Bishop], p.pieces[us][Rook], p.pieces[us][Queen], p.pieces[us][King]
	theirP, theirN, theirB, theirR, theirQ, theirK := p.pieces[them][Pawn], p.pieces[them][Knight], p.pieces[them][Bishop], p.pieces[them][Rook], p.pieces[them][Queen], p.pieces[them][King]

	switch pt {
	case Pawn:
		ourP &^= fromBB
	case Knight:
		ourN &^= fromBB
	case Bishop:
		ourB &^= fromBB
	case Rook:
		ourR &^= fromBB
	case Queen:
		ourQ &^= fromBB
	case King:
		ourK &^= fromBB
	}

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
		} else if theirQ&capBB != 0 {
			theirQ &^= capBB
		} else if theirK&capBB != 0 {
			return false
		}
	}

	if m.isPromo() {
		switch m.promoType() {
		case Queen:
			ourQ |= toBB
		case Rook:
			ourR |= toBB
		case Bishop:
			ourB |= toBB
		case Knight:
			ourN |= toBB
		default:
			return false
		}
	} else {
		switch pt {
		case Pawn:
			ourP |= toBB
		case Knight:
			ourN |= toBB
		case Bishop:
			ourB |= toBB
		case Rook:
			ourR |= toBB
		case Queen:
			ourQ |= toBB
		case King:
			ourK |= toBB
		}
	}

	if flags == FlagCastle {
		if to > from {
			ourR &^= sqBB[from+3]
			ourR |= sqBB[from+1]
		} else {
			ourR &^= sqBB[from-4]
			ourR |= sqBB[from-1]
		}
	}

	occ2 := (ourP | ourN | ourB | ourR | ourQ | ourK) | (theirP | theirN | theirB | theirR | theirQ | theirK)
	var kingSq int
	if pt == King {
		kingSq = to
	} else {
		if ourK == 0 {
			return false
		}
		kingSq = bits.TrailingZeros64(uint64(ourK))
	}

	if pawnAttacks[them^1][kingSq]&theirP != 0 {
		return false
	}
	if knightAttacks[kingSq]&theirN != 0 {
		return false
	}
	if kingAttacks[kingSq]&theirK != 0 {
		return false
	}
	if bishopAttacks(kingSq, occ2)&(theirB|theirQ) != 0 {
		return false
	}
	if rookAttacks(kingSq, occ2)&(theirR|theirQ) != 0 {
		return false
	}
	return true
}

func (p *Position) makeMove(m Move) Undo {
	undo := Undo{
		hash:             p.hash,
		castle:           p.castle,
		epSquare:         p.epSquare,
		halfmove:         p.halfmove,
		captured:         -1,
		lastIrreversible: p.lastIrreversible,
		historyPly:       p.historyPly,
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
			if capSq == 0 {
				p.castle &^= 2
			} else if capSq == 7 {
				p.castle &^= 1
			} else if capSq == 56 {
				p.castle &^= 8
			} else if capSq == 63 {
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
	if p.side == White {
		p.fullmove++
	}
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
	p.historyPly = undo.historyPly
	p.lastIrreversible = undo.lastIrreversible
	p.side = us
	p.castle = undo.castle
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove

	if p.side == Black {
		p.fullmove--
	}

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
	p.hash = undo.hash
}

func (p *Position) makeNullMove() Undo {
	undo := Undo{
		hash:             p.hash,
		castle:           p.castle,
		epSquare:         p.epSquare,
		halfmove:         p.halfmove,
		lastIrreversible: p.lastIrreversible,
		historyPly:       p.historyPly,
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
	p.hash = undo.hash
	p.castle = undo.castle
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove
	p.lastIrreversible = undo.lastIrreversible
	p.historyPly = undo.historyPly
	p.side ^= 1
}

func (p *Position) calculateMobilityAndAttacks(side int) (mgScore, egScore int, attackUnits int) {
	us := side
	them := us ^ 1
	empty := ^p.all
	theirKingBB := p.pieces[them][King]
	theirKingSq := bits.TrailingZeros64(uint64(theirKingBB))
	theirKingZone := kingZoneMask[them][theirKingSq]

	for bb := p.pieces[us][Knight]; bb != 0; {
		from := popLSB(&bb)
		attacks := knightAttacks[from]
		mobility := attacks & empty
		count := bits.OnesCount64(uint64(mobility))
		mgScore += knightMobilityMG[count]
		egScore += mobilityEG[count]
		kingZoneAttacks := attacks & theirKingZone
		attackUnits += KnightAttackWeight * bits.OnesCount64(uint64(kingZoneAttacks))
	}

	for bb := p.pieces[us][Bishop]; bb != 0; {
		from := popLSB(&bb)
		attacks := bishopAttacks(from, p.all)

		mobility := attacks & empty
		count := bits.OnesCount64(uint64(mobility))
		mgScore += bishopMobilityMG[count]
		egScore += mobilityEG[count]

		kingZoneAttacks := attacks & theirKingZone
		attackUnits += BishopAttackWeight * bits.OnesCount64(uint64(kingZoneAttacks))
	}

	for bb := p.pieces[us][Rook]; bb != 0; {
		from := popLSB(&bb)
		attacks := rookAttacks(from, p.all)

		mobility := attacks & empty
		count := bits.OnesCount64(uint64(mobility))
		mgScore += rookMobilityMG[count]
		egScore += mobilityEG[count]

		kingZoneAttacks := attacks & theirKingZone
		attackUnits += RookAttackWeight * bits.OnesCount64(uint64(kingZoneAttacks))
	}

	for bb := p.pieces[us][Queen]; bb != 0; {
		from := popLSB(&bb)
		attacks := rookAttacks(from, p.all) | bishopAttacks(from, p.all)

		mobility := attacks & empty
		count := bits.OnesCount64(uint64(mobility))
		mgScore += queenMobilityMG[count]
		egScore += mobilityEG[count]

		kingZoneAttacks := attacks & theirKingZone
		attackUnits += QueenAttackWeight * bits.OnesCount64(uint64(kingZoneAttacks))
	}

	return mgScore, egScore, attackUnits
}

func (p *Position) evaluateKingSafety(attackUnits int) int {
	return -safetyTable[max(0, min(attackUnits, SafetyTableSize-1))]
}

func northFill(bb Bitboard) Bitboard {
	bb |= bb << 8
	bb |= bb << 16
	bb |= bb << 32
	return bb
}

func southFill(bb Bitboard) Bitboard {
	bb |= bb >> 8
	bb |= bb >> 16
	bb |= bb >> 32
	return bb
}

func (p *Position) evaluatePassedPawns(side int) (mgScore, egScore int) {
	us := side
	them := us ^ 1
	ourPawns := p.pieces[us][Pawn]
	if ourPawns == 0 {
		return 0, 0
	}
	theirPawns := p.pieces[them][Pawn]

	var passedPawns Bitboard
	if us == White {
		frontSpans := southFill(theirPawns)
		adjacentFileMasks := ((theirPawns >> 1) &^ fileMask[7]) | ((theirPawns << 1) &^ fileMask[0])
		adjacentFrontSpans := southFill(adjacentFileMasks)
		passedPawns = ourPawns &^ frontSpans &^ adjacentFrontSpans
	} else {
		frontSpans := northFill(theirPawns)
		adjacentFileMasks := ((theirPawns >> 1) &^ fileMask[7]) | ((theirPawns << 1) &^ fileMask[0])
		adjacentFrontSpans := northFill(adjacentFileMasks)
		passedPawns = ourPawns &^ frontSpans &^ adjacentFrontSpans
	}

	for bb := passedPawns; bb != 0; {
		sq := popLSB(&bb)
		rank := sq / 8
		if us == Black {
			rank = 7 - rank
		}
		mgScore += passedPawnBonusMG[rank]
		egScore += passedPawnBonusEG[rank]
	}
	return
}

func (p *Position) evaluate() int {
	a := p.material[White] - p.material[Black]
	b := p.psqScore[White] - p.psqScore[Black]
	c := p.psqScoreEG[White] - p.psqScoreEG[Black]

	wMobMG, wMobEG, wAttackUnits := p.calculateMobilityAndAttacks(White)
	bMobMG, bMobEG, bAttackUnits := p.calculateMobilityAndAttacks(Black)

	mobilityMG := wMobMG - bMobMG
	mobilityEG := wMobEG - bMobEG

	wPassMG, wPassEG := p.evaluatePassedPawns(White)
	bPassMG, bPassEG := p.evaluatePassedPawns(Black)

	passedMG := wPassMG - bPassMG
	passedEG := wPassEG - bPassEG

	wKingSafety := p.evaluateKingSafety(bAttackUnits)
	bKingSafety := p.evaluateKingSafety(wAttackUnits)
	kingSafety := wKingSafety - bKingSafety

	mg := a + b + mobilityMG + passedMG + kingSafety
	eg := a + c + mobilityEG + passedEG

	ph := p.computePhase()
	phaseScaled := phaseScale(ph)
	score := eg + ((mg-eg)*phaseScaled)/PhaseScale

	if p.side == Black {
		return -score
	}
	return score
}

func (p *Position) orderMoves(moves []Move, bestMove, killer1, killer2 Move) []Move {
	n := len(moves)
	if n <= 1 {
		return moves
	}

	var stackScores [256]int
	scores := stackScores[:n]

	side := p.side

	for i := 0; i < n; i++ {
		m := moves[i]
		score := 0

		if m == bestMove {
			score = scoreHash
		} else {
			isPromo := m.isPromo()
			isCapture := m.isCapture()
			if isPromo {
				score = scorePromoBase + pieceValues[m.promoType()]
			} else if isCapture {
				from := m.from()
				to := m.to()
				capSq := to
				if m.flags() == FlagEP {
					if side == White {
						capSq = to - 8
					} else {
						capSq = to + 8
					}
				}
				attVal := p.square[from]
				vVal := p.square[capSq]
				if attVal >= 0 && vVal >= 0 {
					attacker := attVal & 7
					victim := vVal & 7
					score = scoreCaptureBase + mvvLvaScore(attacker, victim)
				} else {
					score = scoreFallbackCapture
				}
			} else {
				switch m {
				case killer1:
					score = scoreKiller1
				case killer2:
					score = scoreKiller2
				default:
					score = 0
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

func (p *Position) quiesce(alpha, beta, ply int, tc *TimeControl) int {
	p.localNodes++

	if ply <= 1 || (p.localNodes&NodeCheckMaskSearch) == 0 {
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
	var moves []Move
	if inCheck {
		n := p.generateMovesTo(movesArr[:], false)
		moves = movesArr[:n]
	} else {
		n := p.generateMovesTo(movesArr[:], true)
		moves = movesArr[:n]
	}
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
		score := -Mate + ply
		return score
	}

	return best
}

func (p *Position) negamax(depth, alpha, beta, ply int, pv *[]Move, tc *TimeControl, ss *[MaxDepth + 100]SearchStack) int {

	p.localNodes++

	if ply <= 1 || (p.localNodes&NodeCheckMaskSearch) == 0 {
		if tc.shouldStop() {
			return alpha
		}
	}

	if depth <= 0 {
		return p.quiesce(alpha, beta, ply, tc)
	}

	inCheck := p.inCheck()

	origAlpha := alpha

	var hashMove Move
	if e, found, usable := tt.Probe(p.hash, depth); found {
		move, score, _, _, flag := e.unpack()

		if move != 0 {
			hashMove = Move(move)
		}

		if hashMove != 0 && !p.isLegal(hashMove) {
			hashMove = 0
		}

		scoreFromTT := int(score)
		isMateScore := scoreFromTT > MateValue-MaxDepth || scoreFromTT < -MateValue+MaxDepth
		if isMateScore {
			if scoreFromTT > 0 {
				scoreFromTT -= ply
			} else {
				scoreFromTT += ply
			}
			if flag != ttFlagExact {
				usable = false
			}
		}

		if usable {
			switch flag {
			case ttFlagExact:
				if pv != nil && hashMove != 0 {
					*pv = (*pv)[:0]
					*pv = append(*pv, hashMove)
				}
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

	if depth <= 2 && pv == nil && !inCheck && hashMove == 0 && alpha > -Mate+MateScoreGuard && alpha < Mate-MateScoreGuard {
		eval := p.evaluate()
		if depth == 2 {
			if eval <= alpha-Razor2 {
				if v := p.negamax(depth-1, alpha-1, alpha, ply+1, nil, tc, ss); v < alpha {
					return v
				}
			}
		} else {
			if eval <= alpha-Razor1 {
				if v := p.quiesce(alpha-1, alpha, ply+1, tc); v < alpha {
					return v
				}
			}
		}
	}

	if depth >= 3 && !inCheck && !p.isEndgame() && pv == nil {
		R := nullMoveReduction(depth)

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
	moveNum := 0
	var bestChildPV []Move

	if pv != nil {
		*pv = (*pv)[:0]
	}
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
		pvPtr = nil
		legalMoves++
		moveNum++

		if !pvNode && depth <= LMPMaxDepth && legalMoves > LMPMoveThreshold && !inCheck && !m.isCapture() {
			staticEval := p.evaluate()
			if staticEval < alpha-MarginLMP*depth {
				return p.quiesce(alpha, beta, ply+1, tc)
			}
		}

		undo := p.makeMove(m)
		childPV := childPVBuf[:0]
		if pvNode && legalMoves == 1 {
			pvPtr = &childPV
		}
		var score int
		if p.isDraw() {
			score = 0
		} else {
			gaveCheck := p.inCheck()
			childDepth := depth - 1
			k := &ss[ply]
			isKiller := m == k.killer1 || m == k.killer2
			canReduce := childDepth >= LMRMinChildDepth &&
				!inCheck && !gaveCheck &&
				!m.isCapture() && !m.isPromo() &&
				m != hashMove && moveNum > LMRLateMoveAfter &&
				!pvNode && !isKiller
			if canReduce {
				mm := min(moveNum, maxLMRMoves)
				d := min(childDepth, len(lmrTable)-1)
				red := lmrTable[d][mm]

				eff := childDepth - red
				if eff < 1 {
					score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss)
				} else {
					score = -p.negamax(eff, -alpha-1, -alpha, ply+1, nil, tc, ss)
					if score > alpha {
						score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss)
					}
				}
			} else {
				score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss)
			}
		}

		p.unmakeMove(m, undo)

		if score >= beta {
			if !inCheck && !m.isCapture() && !m.isPromo() && m != hashMove {
				k := &ss[ply]
				if m != k.killer1 {
					k.killer2, k.killer1 = k.killer1, m
				}
			}

			if pvNode {
				*pv = append(append((*pv)[:0], m), childPV...)
			}
			storeScore := score
			if storeScore > MateValue-MaxDepth {
				storeScore += ply
			} else if storeScore < -MateValue+MaxDepth {
				storeScore -= ply
			}
			tt.Save(p.hash, m, storeScore, depth, ttFlagLower)
			return beta
		}

		if score > bestScore {
			bestScore = score
			bestMove = m
			if pvNode {
				bestChildPV = append(bestChildPV[:0], childPV...)
			}
		}

		if score > alpha {
			alpha = score
			if pv != nil {
				*pv = append(append((*pv)[:0], m), childPV...)
			}
		}
	}

	if legalMoves == 0 {
		if inCheck {
			score := -Mate + ply
			return score
		}
		return 0
	}

	if pv != nil && len(*pv) == 0 && bestMove != 0 {
		*pv = (*pv)[:0]
		*pv = append(*pv, bestMove)
		*pv = append(*pv, bestChildPV...)
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

func (p *Position) search(tc *TimeControl) Move {
	var bestMove Move
	var ss [MaxDepth + 100]SearchStack

	atomic.StoreInt32(&tc.stopped, 0)

	maxDepth := tc.depth
	if maxDepth == 0 || tc.infinite {
		maxDepth = MaxDepth
	}

	var prevScore int
	var havePrev bool
	var stableBestMove Move
	var pvBuf [MaxDepth]Move
	for depth := 1; depth <= maxDepth; depth++ {

		p.localNodes = 0

		start := time.Now()

		pv := pvBuf[:0]
		var score int

		needFull := false
		if depth >= AspirationStartDepth && havePrev {
			base := prevScore

			if abs(base) >= MateLikeThreshold {
				needFull = true
			} else {

				window := AspirationBase + depth*AspirationStep
				low := base - window
				high := base + window

				score = p.negamax(depth, low, high, 0, &pv, tc, &ss)
				if score <= low || score >= high {
					needFull = true
				}
			}
		} else {
			needFull = true
		}
		if needFull {
			pv = pvBuf[:0]
			score = p.negamax(depth, -Infinity, Infinity, 0, &pv, tc, &ss)
		}

		if !tc.shouldStop() {
			prevScore = score
			havePrev = true
		}

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

		if atomic.LoadInt32(&tc.stopped) == 0 && bestMove != 0 {
			stableBestMove = bestMove
		}

		absScore := score
		if absScore < 0 {
			absScore = -absScore
		}

		if absScore >= MateValue-MaxDepth && bestMove != 0 {
			stableBestMove = bestMove
			break
		}

		if absScore >= Mate-MateScoreGuard {
			matePly := Mate - absScore
			mateMoves := (matePly + 1) / 2
			if mateMoves < 1 {
				mateMoves = 1
			}
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

		if absScore >= Mate-MateScoreGuard {
			if bestMove != 0 && p.isLegal(bestMove) {
				return bestMove
			}
			break
		}

		if tc.shouldStop() || !tc.shouldContinue(elapsed) {
			break
		}
	}

	if bestMove != 0 && p.isLegal(bestMove) {
		return bestMove
	}
	if stableBestMove != 0 && p.isLegal(stableBestMove) {
		return stableBestMove
	}

	fmt.Fprintln(os.Stderr, "# Warning: search could not find a legal move to play.")
	return 0
}

func (tc *TimeControl) Stop() {
	atomic.StoreInt32(&tc.stopped, 1)
}

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

	baseMs := min(fromBank+myInc/perMoveCapDiv, usableTime)
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
	if d.IsZero() {
		return false
	}
	return time.Until(d) <= 0
}

func (tc *TimeControl) shouldContinue(lastIter time.Duration) bool {
	if atomic.LoadInt32(&tc.stopped) != 0 {
		return false
	}

	if tc.infinite || tc.depth > 0 || lastIter <= 0 {
		return true
	}

	if tc.deadline.IsZero() {
		return true
	}
	remain := time.Until(tc.deadline)
	if remain <= 0 {
		return false
	}

	minRequired := lastIter*nextIterMult + continueMargin
	return remain > minRequired
}

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
	if move == 0 {
		fmt.Println("bestmove 0000")
	} else {
		fmt.Println("bestmove", move)
	}
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
		if nameStart >= len(parts) {
			return "", ""
		}
		return strings.Join(parts[nameStart:], " "), ""
	}

	if nameStart >= nameEnd {
		return "", ""
	}
	name = strings.Join(parts[nameStart:nameEnd], " ")

	if valueStart >= len(parts) {
		return name, ""
	}
	value = strings.Join(parts[valueStart:], " ")
	return name, value
}

func uciLoop() {
	pos := NewPosition()

	var cmdMutex sync.Mutex

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	fmt.Fprintln(os.Stderr, "# Soomi V1.1 ready. Type 'help' for available commands.")

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		cmd := parts[0]

		switch cmd {
		case "uci":
			fmt.Println("id name Soomi V1.1")
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
				cmdMutex.Lock()
				InitTT(sizeMB)
				cmdMutex.Unlock()
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

			if parts[1] != "startpos" {
				fmt.Println("info string only 'position startpos [moves ...]' is supported; resetting to startpos")
				pos.setStartPos()
				break
			}
			pos.setStartPos()

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
						piece := ".PNBRQK"[pt+1]
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
	fmt.Println(`# Soomi V1.1 - Available Commands:

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
	fmt.Fprintln(os.Stderr, "Soomi V1.1 - UCI Chess Engine")
	fmt.Fprintln(os.Stderr, "Type 'help' for available commands or 'uci' to enter UCI mode")
	fmt.Fprintln(os.Stderr)
	uciLoop()
}

// To make an executable
// go build -trimpath -ldflags "-s -w" -o Soomi-V1.1.exe soomi.go
