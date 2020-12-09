# 50.007-design-project

Designing sequence labelling model for informal texts using Hidden Markov Model (HMM).

Authors:

- Loh De Rong
- Tiong Shan Kai
- Koh Ting Yew

# Instructions to run the code



## Part 2

```bash
cd part2
python3 hmm_part2.py <dataset-name>
```

Where `<dataset-name>` is any of the following: `EN`, `SN`, `CN`. 

## Part 3

```bash
cd part3
python3 hmm_part3.py <dataset-name>
```

Where `<dataset-name>` is any of the following: `EN`, `SN`, `CN`. 

## Part 4

```bash
cd part4
python3 hmm_part4.py EN
```

## Part 5 - Design Challenge

```bash
cd part5
edit this shan kai
```

## EvalScript

```
cd EvalScript
python3 EvalResult.py "../dataset/<dataset-name>/dev.out" "../dataset/<dataset-name>/<prediction-output>"
```

Where `<dataset-name>` is any of the following: `EN`, `SN`, `CN`. 

`<prediction-output>` is any of the following: `dev.p2.out`, `dev.p3.out`, `dev.p4.out`, `dev.p5.out`.