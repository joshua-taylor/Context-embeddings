## 1. Word associations (nearest token meanings)

- **river**: rivers, tribut, valley, basin, lake, nile, danube, flows
- **doctor**: degree, degrees, bachel, graduate, ด, doctors, master, อ
- **guitar**: bass, guitars, acoustic, instrument, instruments, string, playing, piano
- **happy**: you, i'm, happ, luck, it's, i', want
- **london**: edinburgh, manchester, oxford, liverpool, york, cambridge, croyd
- **war**: wars, occupation, invasion, battle, civil, conflict, during, napole
- **money**: funds, cash, loans, pay, credit, payment, dollars, payments
- **running**: run, ran, operating, runs, longest, processors, compatible, windows

## 2. Fill in the blank (cloze stream, top guesses)

- she sat on the river ___ and watched the water  →  ,, river, as, sea, ##'s, land, valley, ##ous
- he deposited the money at the ___ on monday  →  time, on, end, age, next, home, in, his
- the ___ barked at the postman  →  term, aclu, name, first, american, ,, for, near
- the doctor gave the patient some ___  →  problems, form, time, others, al, life, u, friends
- paris is the capital of ___  →  azerbaijan, europe, england, amsterdam, france, world, egypt, albania
- the band played a new ___ at the concert  →  york, city, constitution, states, at, film, and, angeles
- the war ended in ___  →  2008, 2005, 2000, 2009, afghanistan, –, 1991, 1998
- she was elected ___ of the united states  →  president, to, member, governor, council, vice, minister, as

## 3. What comes next (forward stream)

- the river flows into the …  →  river, north, amazon, south, atlantic, into, water, coast
- she was born in …  →  paris, the, com, cong, 117, munich, london, athens
- the album was released …  →  on, in, ,, by, except, alongside, april, march
- the united states …  →  states, president, navy, census, ,, ruling, kingdom, department
- the battle of …  →  marath, bad, l, war, t, white, battle, pan

Sampled continuations (sampled from the top 4 guesses at each step):

- the city is area area 510 bc ) * 1799 – 13 – 6 )
- the river god , andtowers the end of the original home of the
- in the early years of the warhol's death in 1833 – 154 bc ) * 1776
- the band members by partyinas .

## 4. One word, different senses (contextual vectors, meaning excluded)

**bank** (cosine; rows = sentences)

| | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| s1: she sat on the river bank and watched the water | 1.00 | 0.63 | 0.50 | 0.54 |
| s2: the boat drifted towards the muddy bank of the stream | 0.63 | 1.00 | 0.47 | 0.51 |
| s3: the bank raised its interest rates again | 0.50 | 0.47 | 1.00 | 0.61 |
| s4: he borrowed money from the bank to buy a house | 0.54 | 0.51 | 0.61 | 1.00 |

what fits in that slot (cloze guesses with the word hidden):

- she sat on the river ___ and watched the water  →  ,, river, as, sea, ##'s, land
- the boat drifted towards the muddy ___ of the stream  →  part, sea, coast, side, ##ness, because
- the ___ raised its interest rates again  →  has, ag, apple, aclu, company, government
- he borrowed money from the ___ to buy a house  →  ability, order, in, money, right, back

**plant** (cosine; rows = sentences)

| | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| s1: the plant grew tall in the sunny garden | 1.00 | 0.55 | 0.46 | 0.44 |
| s2: water the plant every day and it will flower | 0.55 | 1.00 | 0.56 | 0.56 |
| s3: the car plant employs two thousand workers | 0.46 | 0.56 | 1.00 | 0.47 |
| s4: the power plant was closed after the accident | 0.44 | 0.56 | 0.47 | 1.00 |

what fits in that slot (cloze guesses with the word hidden):

- the ___ grew tall in the sunny garden  →  population, city, most, largest, town, war
- water the ___ every day and it will flower  →  water, next, city, of, almost, same
- the car ___ employs two thousand workers  →  which, ##pent, racing, ##box, ##p, ##ine
- the power ___ was closed after the accident  →  of, at, where, area, and, which

**bass** (cosine; rows = sentences)

| | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| s1: he caught a large bass in the lake | 1.00 | 0.60 | 0.49 | 0.57 |
| s2: the fishermen landed bass and trout | 0.60 | 1.00 | 0.50 | 0.52 |
| s3: he plays bass guitar in a rock band | 0.49 | 0.50 | 1.00 | 0.59 |
| s4: the song has a heavy bass line | 0.57 | 0.52 | 0.59 | 1.00 |

what fits in that slot (cloze guesses with the word hidden):

- he caught a large ___ in the lake  →  part, role, scale, of, army, result
- the fishermen landed ___ and trout  →  ,, -, sea, the, were, ##men
- he plays ___ guitar in a rock band  →  a, the, ., with, on, acoustic
- the song has a heavy ___ line  →  metals, united, the, number, tree, same

## 5. Word-in-context retrieval: nearest corpus occurrences to a query use

- query: *she sat on the river bank and watched the water*
    - to the southeast lies morant bank , with the morant cays , from morant point , the easternmost point of mainland jamaica .
    - in mid - 1950 , jordan formally annexed the west bank and jericho residents , like other residents of west bank localities became jordanian citizens .
    - he worked for the molsons bank on stanley street in montreal , and then for the engineering firm m .
- query: *the bank raised its interest rates again*
    - the central bank buys us dollars at 0 .
    - the bank was the first local publicly traded corporation in the gulf .
    - he was sent to work in jos , nigeria , by the bank in 1967 and he nearly died in a car accident there .
- query: *the plant grew tall in the sunny garden*
    - the plant was first recorded in america in 1624 , but it was named after pehr kalm , who sent samples to linnaeus in the 18th century .
    - the plant is naturally found on rocky slopes and mountainous forest areas .
    - the plant was originally brought to europe as an ornamental plant during the 18th century .
- query: *the power plant was closed after the accident*
    - the main frames were bolted onto the power plant in its top quarter .
    - the fuel was injected via a pump from the tanks to the power plant .
    - the flight report , by hauptmann willy neuenhofen , stated the only problem was with the small radiator , which caused the power plant to overheat .

## 6. Sentence similarity (whitened sentence vectors): closest match in the pool

- a man is playing a guitar  →  a man plays the guitar on stage  (0.50)
- a man plays the guitar on stage  →  a man is playing a guitar  (0.50)
- a woman is slicing an onion  →  a man plays the guitar on stage  (0.08)
- someone is cutting vegetables in the kitchen  →  a pair of dogs run across the grass  (0.08)
- the stock market fell sharply today  →  share prices dropped steeply this morning  (0.35)
- share prices dropped steeply this morning  →  the stock market fell sharply today  (0.35)
- the doctor examined the patient  →  the physician checked the sick man  (0.18)
- the physician checked the sick man  →  the doctor examined the patient  (0.18)
- two dogs are running in a field  →  a pair of dogs run across the grass  (0.37)
- a pair of dogs run across the grass  →  two dogs are running in a field  (0.37)
- the government announced new taxes  →  the parliament approved a new tax law  (0.31)
- the parliament approved a new tax law  →  the government announced new taxes  (0.31)
