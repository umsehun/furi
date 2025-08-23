import random, statistics, argparse
from collections import Counter

def clamp(x, lo, hi): return max(lo, min(hi, x))

def get_tier(a):
    if a < 20: return 'cold'
    if a < 40: return 'awkward'
    if a < 60: return 'friend'
    if a < 80: return 'close'
    return 'lover'

# 너가 정한 규칙
TIER_SUCCESS = {'cold':-40,'awkward':-20,'friend':10,'close':25,'lover':40}
TIER_UP_MULT = {'cold':1.30,'awkward':1.20,'friend':1.00,'close':0.90,'lover':0.60}
TIER_DN_MULT = {'cold':1.50,'awkward':1.20,'friend':1.00,'close':0.60,'lover':0.60}

TAGS = ['kind','humor','smart','bold']

def softcap(a):
    # 70 이상 소프트캡 (A-70)*0.2
    return a if a < 70 else a - (a-70)*0.2

def success_rate(a):
    tier = get_tier(a)
    base = 50
    difficulty = random.uniform(0, 30)   # 이벤트 난이도
    context    = random.uniform(-10, 10) # 장소/시간 보정
    luck       = random.uniform(-5, 5)   # 소폭 운빨
    r = base + TIER_SUCCESS[tier] - difficulty + context + luck
    return clamp(r, 5, 95)

def repeat_penalty(cnt):
    if cnt <= 2: return 1.0
    if cnt == 3: return 0.9
    if cnt == 4: return 0.8
    return 0.7

def random_weight(tag, rep_cnt):
    # 캐릭터 선호(0.8~1.2), 컨텍스트(0.9~1.2), 스탯(0~5 → +3%/스탯), 난이도/랜덤
    char_pref = random.uniform(0.8, 1.2)
    ctx       = random.uniform(0.9, 1.2)
    stat      = 1 + random.randint(0,5) * 0.03
    diff_adj  = random.uniform(0.95, 1.05)
    rng       = random.uniform(0.95, 1.05)
    rep       = repeat_penalty(rep_cnt)
    return clamp(char_pref*ctx*stat*diff_adj*rng*rep, 0.5, 1.6)

def step(aff, day_gain_used, rep_state):
    # 선택 품질: +3(45%), 0(35%), -2(20%)
    r = random.random()
    base = 3 if r < 0.45 else (0 if r < 0.80 else -2)

    tag = random.choice(TAGS)
    rep_state[tag] = rep_state.get(tag,0)+1
    for t in TAGS:
        if t != tag and rep_state.get(t,0) > 0:
            rep_state[t] = max(0, rep_state[t]-1)

    sr = success_rate(aff)
    ok = (random.uniform(0,100) <= sr)
    w  = random_weight(tag, rep_state[tag])

    decay = aff/200.0
    cap   = 8
    tier  = get_tier(aff)

    if ok:
        raw = base*w - decay
        delta = clamp(raw, -cap, cap) * TIER_UP_MULT[tier]
        # 일일 상한: 80+ 구간 +12, 그 외 +20
        daily_cap = 12 if aff >= 80 else 20
        can_gain = max(0.0, daily_cap - day_gain_used)
        gain = max(0.0, delta)
        if gain > can_gain:
            delta -= (gain - can_gain)
        day_gain_used += max(0.0, delta)
    else:
        base_fail = -3.0 + (-1.5 if base < 0 else 0.0)  # 나쁜 선택 실패 시 추가 하락
        raw = base_fail * w * random.uniform(0.95,1.05)
        delta = clamp(raw, -6, -1) * TIER_DN_MULT[tier]

    a2 = clamp(softcap(aff + delta), 0, 100)
    ethical = (a2 >= 95)
    return a2, day_gain_used, ethical, ok, sr, delta, tag

def run_session(days=10, turns=6, a0=None):
    a = random.uniform(20,30) if a0 is None else a0
    reached = set([get_tier(a)])
    ethical = False
    history = []
    for _ in range(days):
        day_gain = 0.0
        rep = {}
        for _ in range(turns):
            a, day_gain, eth, ok, sr, delta, tag = step(a, day_gain, rep)
            ethical = ethical or eth
            reached.add(get_tier(a))
            history.append((a, ok, sr, delta, tag))
    return a, reached, ethical, history

def simulate(N=1000, seed=None, days=10, turns=6):
    if seed is not None:
        random.seed(seed)
    finals=[]; reach=Counter(); ethical_cnt=0; deltas=[]
    for _ in range(N):
        a, reached, eth, hist = run_session(days, turns)
        finals.append(a)
        ethical_cnt += 1 if eth else 0
        for tr in reached: reach[tr]+=1
        deltas.extend([h[3] for h in hist])

    finals.sort()
    def pct(p):
        i = int(p*(len(finals)-1))
        return finals[i]
    rep = {
        "N": N,
        "avg_final": round(statistics.mean(finals),3),
        "median_final": round(statistics.median(finals),3),
        "p25_final": round(pct(0.25),3),
        "p75_final": round(pct(0.75),3),
        "p95_final": round(pct(0.95),3),
        "p99_final": round(pct(0.99),3),
        "reach_rate": {k: round(reach[k]/N,3) for k in ['cold','awkward','friend','close','lover']},
        "ethical_unlock_rate": round(ethical_cnt/N,3),
        "avg_delta": round(statistics.mean(deltas),3)
    }
    return rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("N", nargs="?", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()

    rep = simulate(args.N, args.seed, args.days, args.turns)
    for k,v in rep.items():
        print(k, ":", v)