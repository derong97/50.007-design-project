# 50.007-design-project

Designing sequence labelling model for informal texts using Hidden Markov Model (HMM).

Authors:

- Loh De Rong (1003557)
- Koh Ting Yew (1003339)
- Tiong Shan Kai (1003469)



The predictions from each of the parts 2 to 5 are all located in the `dataset` directory. 

For a more in-depth description and breakdown of our implementation for this project, you may refer to `Report.pdf`.

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

## Part 5 - Design Challenge - edit this shan kai

```bash
cd part5
edit this shan kai
```

## EvalScript

We have evaluated our prediction outputs for Parts 2 to 4. The evaluation results can be located in our Jupyter notebooks. 

To perform the evaluation yourselves: run the following:

```bash
cd EvalScript
python3 EvalResult.py "../dataset/<dataset-name>/dev.out" "../dataset/<dataset-name>/<prediction-output>"
```

`<dataset-name>` is any of the following: `EN`, `SN`, `CN`. 

`<prediction-output>` is any of the following: `dev.p2.out`, `dev.p3.out`, `dev.p4.out`, `dev.p5.out`.

<br>

<hr>





