# Genetic Cryptarithmetic Solver

## Overview

This project implements a genetic algorithm to solve a Cryptarithmetic puzzle where each letter represents a digit, and the goal is to find a digit assignment that satisfies the equation. The genetic algorithm uses a population-based approach to evolve potential solutions over generations.

## Requirements

- Python 3.x
- NumPy library

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/phdangg/cryptarithmetic_genetic_search.git
   ```

2. Navigate to the project directory:

   ```bash
   cd cryptarithmetic_genetic_search
   ```

3. Run the solver for different levels:

   ```bash
   python main.py
   ```

   The solver will process input files in the `level_1`, `level_2`, and `level_3` folders and output the results.

## Input Format

The input equations should be provided in a specific format:

- Use uppercase letters (A-Z) for variables.
- Operators allowed: +, -, *, =.
- Parentheses are allowed for nested expressions.

Example:

   ```plaintext
   SEND + MORE = MONEY
   ```

## Output

The program will output the solution to each equation or indicate if there is no solution. The results will be stored in output files with the prefix "output" in the same directory as the input files.

## Examples

Check the `level_1`, `level_2`, and `level_3` folders for example input files and corresponding output files.
