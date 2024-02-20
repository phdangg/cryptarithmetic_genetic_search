import random
import time
import numpy as np
import re
import os
DOMAIN_RANGE = 10 # (0,1,2,3,4,5,6,7,8,9)
POPULATION_SIZE = 50
TIME_LIMIT = 60*60 # 1 hour

def create_eval_expression(chromosome,word):
    """ 
    letter: SEND
    SEND: 1000*S + 100*E + 10*N 1*D
    evaluation_expression is matching the index of chromosome for each letter
    and assign the decimal value of each variable according to its position in the term
    chromesome =        ['D', 'E', 'M', 'N', 'O', 'R', 'S', 'Y', '-', '-']
    evaluation_expression = [ 1,  100,  0,   10,  0,   0, 1000,   0,   0,   0]
    """

    '''make geometric series with a common ratio of 10, start term is 
    letters from the last position.
    We are reversing the word for this purpose
    '''
    word = word[::-1]

    # create an empty expression
    eval_vec = [0]*DOMAIN_RANGE

    for i in range(len(word)):
        j = chromosome.index(word[i]) 
        # j is the index of letter (position 'i' of word) in chromosome 
        if eval_vec[j] != 0:
            eval_vec[j] += 10 ** i
        else:
            eval_vec[j] = 10 ** i
    return eval_vec

def extract_words(expression):

    # Find all the words in the expression (including those inside parentheses)
    words = re.findall(r'\b[A-Z]+\b', expression)

    return words

def encode(input):
    word_list = extract_words(input)
    chromosome = []
    for word in word_list:
        chromosome.extend(list(word))
    chromosome = list(set(chromosome))
    chromosome.sort() # Sorting this for the same problem evaluation expression
    UNSIGNED_COLUMN = DOMAIN_RANGE - len(chromosome)
    chromosome.extend("-"*UNSIGNED_COLUMN)
    return chromosome

#=============================GENETIC SEARCH======================================================================================
def generate_pop():
    population = [i % DOMAIN_RANGE for i in range(POPULATION_SIZE * DOMAIN_RANGE)]
    population = [population[i:i+DOMAIN_RANGE] for i in range(0, len(population), DOMAIN_RANGE)]
    
    for i in range(len(population)):
        random.shuffle(population[i])
    
    return population
def mutate(population):
    """ mutate the population """
    # mutate by swapping two randomly selected column in the whole population matrix

    # randomly select columns
    col1 = random.randint(0, DOMAIN_RANGE-1)
    col2 = random.randint(0, DOMAIN_RANGE-1)
    for row in population:
        row[col1], row[col2] = row[col2], row[col1]
    return population
def fitness(population,eval_vector):
    matrix = np.abs(np.matmul(population, eval_vector))
    return matrix

def has_solution(pop_fitness):
    return any(value ==0 for value in pop_fitness)

def chromosome_mapping(chromosome,state):
    '''For printing solution'''
    result  = []
    for letter, num in zip(chromosome, state):
        if letter != "-": 
         string_result = letter + "=" + str(num)
         result.append(string_result)
    return result


def solution(pop,pop_fitness):
    index = np.where(pop_fitness == 0)[0][0]
    sol = pop[index]
    return sol

def compare_eval(eval1, eval2):
    '''Compare fitness of 2 population, if population 1 has a greater fitness value than population 2
    return True, otherwise False'''
    return [True if e1 < e2 else False for e1, e2 in zip(eval1, eval2)]

def reproduce(pop_1,pop_1_fitness,pop_2,pop_2_fitness):
    '''
    selection between the two populations based on the fitness values 
    Chosing element of pop_1 if fitness is "True", vice versa
    '''
    fitness = compare_eval(pop_1_fitness, pop_2_fitness)
    pop_1 = [pop_1[i] if fit else pop_2[i] for i, fit in enumerate(fitness)]
    return pop_1

def left_most_letter_value(pop,index_of_left_most_letter):
    for index in index_of_left_most_letter:
        if pop[index] == 0:
            return 0
    return 1
def get_index_of_left_most_letter(chromosome,lst_words):
    first_letter = set()
    for word in lst_words:
        first_letter.add(word[0])
    first_letter = sorted(first_letter)

    index = []
    for letter in first_letter:
        index.append(chromosome.index(letter))
    return index

def genetic_search(input_string):
    #Make sure the equation like "SEND+MORE=MONEY" not "MONEY=SEND+MORE"
    parts = input_string.split('=')
    # Remove leading and trailing whitespaces from each part
    parts = [part.strip() for part in parts]
    pattern = r'[+\-*/()]'
    match = re.search(pattern, parts[1])
    if match:
        input_string = f"{parts[1]}={parts[0]}"
    
    extracted_input = extract_nested_expression(input_string)
    chromosome = encode(input_string)
    
    right_side_expression = evaluate_expression(extracted_input[:-2],chromosome)
    left_side_expression = create_eval_expression(chromosome,extracted_input[-1])
    eval_vector = [x-y for x,y in zip(left_side_expression,right_side_expression)]

    index_of_left_most_letter = get_index_of_left_most_letter(chromosome,extract_words(input_string))



    start_time = time.time()
    end_time = start_time + TIME_LIMIT
    pop_1 = pop_2 = generate_pop()

    generation_counter = 0
    while time.time() < end_time:
        pop_1_fitness = fitness(pop_1,eval_vector)
        pop_2_fitness = fitness(pop_2,eval_vector)

        if has_solution(pop_1_fitness):
            sol = solution(pop_1,pop_1_fitness)
            if left_most_letter_value(sol,index_of_left_most_letter) != 0: 
                times = time.time() - start_time
                print("Generation: ",generation_counter)
                print("Time (millisecond): ",times )
                return chromosome_mapping(chromosome,sol)
        elif has_solution(pop_2_fitness):
            sol = solution(pop_2,pop_2_fitness)
            if left_most_letter_value(sol,index_of_left_most_letter) != 0: 
                times = time.time() - start_time
                print("Generation: ",generation_counter)
                print("Time (millisecond): ",times )
                return chromosome_mapping(chromosome,sol)

        pop_1 = reproduce(pop_1,pop_1_fitness,pop_2,pop_2_fitness)

        if generation_counter % 10 == 0: pop_2 = generate_pop()
        else: pop_2 = mutate(pop_1)
        generation_counter += 1
    
    return "NO SOLUTION"

#====================================================================================================================
def evaluate_expression(expression,chromosome):
    stack = []
    for item in expression:
        if isinstance(item, list):
            nested_result = evaluate_expression(item,chromosome)
            stack.append(nested_result)
        else:
            stack.append(item)

    while len(stack) > 1:
        left_operand = stack.pop(0)
        if isinstance(left_operand, list): 
            left_eval = left_operand
        else: 
            left_eval = create_eval_expression(chromosome,left_operand)

        operator = stack.pop(0)

        right_operand = stack.pop(0)
        if isinstance(right_operand, list): right_eval = right_operand
        else:
            right_eval = create_eval_expression(chromosome,right_operand)

        if operator == '+':
            tmp = [x+y for x,y in zip(left_eval,right_eval)]
            stack.insert(0, tmp)
        elif operator == '-':
            tmp = [x-y for x,y in zip(left_eval,right_eval)]
            stack.insert(0, tmp)
        elif operator == '*':
            tmp = [x*y for x,y in zip(left_eval,right_eval)]
            stack.insert(0, tmp)

    return stack[0]

def combine_arrays(array, multiple_index):
    left = array[:multiple_index - 1]
    right = array[multiple_index + 2:]
    combined = [array[multiple_index - 1]] + [array[multiple_index]] + [array[multiple_index + 1]]
    result = left + [combined] + right
    return result

def extract_nested_expression(expression):
    expression = expression.replace(" ", "")
    nested_expression = []
    stack = []
    temp_str = ""

    for char in expression:
        if char == '(':
            if temp_str:
                nested_expression.append(temp_str)
                temp_str = ""
            stack.append(nested_expression)
            nested_expression = []
        elif char == ')':
            if temp_str:
                nested_expression.append(temp_str)
                temp_str = ""
            last_expression = stack.pop()
            last_expression.append(nested_expression)
            nested_expression = last_expression
        elif char in '-+=*':
            if temp_str:
                nested_expression.append(temp_str)
                temp_str = ""
            nested_expression.append(char)
        else:
            temp_str += char

    if temp_str:
        nested_expression.append(temp_str)
    
    len_nested_expression = len(nested_expression)
    i = 0
    while i < len_nested_expression:
        if nested_expression[i] == "*":
            nested_expression = combine_arrays(nested_expression,i)
            len_nested_expression = len(nested_expression)
            continue
        i +=1
    return nested_expression

def is_valid_input(equation):
    # Kiểm tra tính hợp lệ của các ký tự 
    validChars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ+-*()= ')
    if any(char not in validChars for char in equation):
        return False

    # Kiểm tra tính hợp lệ của các ký tự chữ cái không xuất hiện trùng lặp trên cả 3 phần tử
    uniqueChars = set(char for char in equation if char.isalpha())
    if len(uniqueChars) > 10:
        return False

    # Kiểm tra xem dấu '=' chỉ xuất hiện 1 lần
    if equation.count('=') != 1:
        return False

    # Kiểm tra xem dấu '+' và '-' không xuất hiện liên tiếp
    operators = '+-*'

    for i in range(len(equation) - 1):
        if equation[i] in operators and equation[i + 1] in operators:
            return False

    # Kiểm tra tính hợp lệ của cặp dấu ngoặc '(' và ')'
    openParentheses = []
    for char in equation:
        if char == '(':
            openParentheses.append(char)
        elif char == ')':
            if not openParentheses:
                return False
            openParentheses.pop()

    if openParentheses:
        return False

    return True

def output(sol,directory,filename):
    output_filename = "output" + filename[5:]
    f = os.path.join(directory, output_filename)
    file_output = open(f,'w')
    if sol == "NO SOLUTION":
        file_output.writelines(sol)
        return
    sol_str = ""
    for i in sol:
        sol_str += i[2]
    print(sol_str)
    file_output.writelines(sol_str)
    file_output.close()
    return
        
def input_folder_iterate(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            file_input = open(f, "r")
            input_string = (file_input.readline())
            if not is_valid_input(input_string): continue
            print(input_string)
            sol = genetic_search(input_string)
            print("Solution: ",sol)
            output(sol,directory,filename)
            print()
            file_input.close()

if __name__ == "__main__":
    print("PROCESSING LEVEL 1 INPUT")
    input_folder_iterate("level_1")
    print()
    print("===========================")
    print("PROCESSING LEVEL 2 INPUT")
    input_folder_iterate("level_2")
    print()
    print("===========================")
    print("PROCESSING LEVEL 3 INPUT")
    input_folder_iterate("level_3")
    print()

    
    


    