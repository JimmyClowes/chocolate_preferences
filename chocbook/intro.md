# Modelling chocolate preferences from ranking data

## Summary
This project works through an analysis of ranking data relating to relative preferences of a selection of chocolates carried out by a number of people.

It aims to make inferences from the ranking data about:
- which chocolates appear to be systematically preferred at the population level
- whether there are some chocolates that have higher variation in appeal than others

## What is this about?
The data used in this project came about from an evening with a group of friends with a box of Celebrations and a box of Heroes on the table, where someone suggested everyone rank each chocolate from best to worst. Because that's the kind of really cool thing we do.

## Why does this exist?
Fair question. ["Why does Rice play Texas?"](https://en.wikipedia.org/wiki/We_choose_to_go_to_the_Moon) My main areas of professional interest at the moment are in probablistic methods using software like [PyMC](https://www.pymc.io/welcome.html) and [Stan](https://mc-stan.org/), so when seeing the ranking data, the question occurred to me about whether it could be used to understand the appeal of the chcolates at the population level, both in terms of order and in terms of variation. I hadn't previously employed any methods for ordered data using a Bayesian approach, so I knew this would be a challenge and I would learn something.

## What's involved in the project?
The project follows the process of [Modern Statistical Workflow](https://khakieconomics.github.io/2016/08/29/What-is-a-modern-statistical-workflow.html), which is applicable to the development of Bayesian models and aims to ensure models are fit for purpose by first testing them on simulated data so that they can be trusted when applied to real data.

The steps of the modern statistical workflow are worked through on the pages listed in the contents below.

```{tableofcontents}
```
