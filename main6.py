from follow import follow_pos
from state import *
from numpy import nan
from pandas import DataFrame as df, MultiIndex
import traceback

states = []
parsing_table = df()
all_scope = []
my_glo_result = None
back = None

class Node:
    """Base class for AST nodes."""
    def evaluate(self):
        raise NotImplementedError("Subclasses must implement the evaluate method.")

class BinaryOperationNode(Node):
    """Represents a binary operation in the AST."""
    def __init__(self, left, right, operator):
        self.left = left
        self.right = right
        self.operator = operator

    def evaluate(self):
        left_value = ''
        right_value = ''
        if self.left is not None and self.right is not None:
            left_value = self.left.evaluate()
            right_value = self.right.evaluate()

        else:
            if self.left is not None:
                left_value = self.left.evaluate()
            elif self.right is not None:
                right_value = self.right.evaluate()
            else:
                return
                
        
        global back
        # Perform the appropriate operation based on the operator
        if self.operator == '+':
            return left_value + right_value
        elif self.operator == '-':
            if type(left_value) is int and type(right_value) is int:
                return left_value - right_value
            elif type(left_value) is int and type(right_value) is not int:
                right_value = left_value
                left_value = back
                return left_value - right_value
            elif type(left_value) is not int and type(right_value) is int:
                return right_value
            else:
                return 0
        elif self.operator == '*':
            return left_value * right_value
        elif self.operator == '/':
            if right_value == 0:
                raise ZeroDivisionError("Division by zero")
            return left_value / right_value
        else:
            print('ISSUE', self.operator)
        
            #raise ValueError(f"Unknown operator: {self.operator}")

class LiteralNode(Node):
    """Represents an integer literal in the AST."""
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value




class SyntaxAnalyzer:
    def __init__(self):
        pass

    def lexer(self, input_str):
        lexemes = []
        tokens = []
        i = 0
        n = len(input_str)
        
        while i < n:
            if input_str[i].isdigit():
                # If the current character is a digit, form a full integer
                number = input_str[i]
                i += 1
                while i < n and input_str[i].isdigit():
                    number += input_str[i]
                    i += 1
                lexemes.append(int(number))
                tokens.append('N')
            elif input_str[i] in '+-*/':
                # If the current character is an arithmetic operator
                lexemes.append(input_str[i])
                tokens.append(input_str[i])
                i += 1
            else:
                # Unrecognized characters are ignored for simplicity
                i += 1
        
        # Append the end of input marker
        lexemes.append('$')
        tokens.append('$')
        
        return lexemes, tokens



def grammar_fromstr(g):
    grm = g.split('\n')
    rules = []
    for r in grm:
        if r.strip()=='':continue # skip empty lines
        lhs, rhs_ls = r.split('=>')
        for rhs in rhs_ls.split('|'):
            rules.append((lhs.strip(),rhs.strip().split()))
    return rules

def augment(grammar_str):
    #'''
    #- Rules must be in form LHS => RHS
    #- NONTERMINALS must start with (`) and be UPPERCASE. use only [A-Z] and (_) for NONTERMINALS
    #- Terminals must be lower case
    #- If there are many rhs for one lhs, you should put them all in the same line and separate them with (|) i.e: X => Y | Z | !εpslon
    #- Use the form (!εpslon) for writing epslons 
    #'''
    grammar= grammar_fromstr(grammar_str)
    rhs = [grammar[0][0]] # The start symbol
    aug=Rule(grammar[0][0]+"'", tuple(rhs))
    s = State()
    s.add_rule(aug)
    Rule.augmented.append(aug)
    for r in grammar:
        Rule.augmented.append(Rule(r[0],r[1]))
    return s, extract_symbols(grammar)

def extract_symbols(rules):
    terminals = []
    non_terminals = []
    for r in rules:
        if r[0] not in non_terminals:
            non_terminals.append(r[0])
        for s in r[1]:
            if not s.startswith('`'):
                if s not in terminals:
                    if s != '!εpslon':
                        terminals.append(s)
            else:
                if s not in non_terminals:
                    non_terminals.append(s)
    terminals.append('$')
    return (non_terminals, terminals)

def goto_operation():
    #s = states[0]
    for s in states:
        transitions = []
        for r in  s.rules:
            rule = r.movedot()
            dotatend = rule == None
            if dotatend:
                continue
            transitions.append((r.handle(), rule))
        
        # If a new transition item is already in the current state, make a cycle
        gotoself(transitions, s)
        
        for t in transitions:

            # Put all items with the same Transition symbol in one state
            items_same_X = [r for r in transitions if r[0] == t[0]]
            make_transition(s, items_same_X) 
    return State.graph

def gotoself(transitions, s):
    # If a new transition is already in the current state, make a cycle
    for t in transitions:
        if t[1] in s.rules:
            s.goto( s._i, t[0])
            transitions.remove(t)

def make_transition(source, items_same_X):
    new_state = newstate(items_same_X)
    ### exists
    # If new Items I alraedy there in a state s, goto s
    # Else, make a new state Si of I 
    exists = False
    for s in states:
        if new_state == s:
            source.goto(s._i, symbol=items_same_X[0][0])
            exists=True
            State._count = State._count-1
            break
    ###
    if not exists:
        new_state.closure()
        states.append(new_state)
        source.goto(new_state._i, symbol=items_same_X[0][0])

def newstate(items_same_X):
    new_state = State()
    for r in items_same_X:
        new_state.add_rule(r[1])
    return new_state

def parsing_table_skelton(non_terminals, terminals):
    levels = (['action']*len(terminals) + ['goto']*len(non_terminals))
    columns = terminals+non_terminals
    index = [s._i for s in states]
    return df(index=index, columns=MultiIndex.from_tuples(list(zip(levels,columns)),names=['table','state'])).fillna('_')

def slr_parsing_table(items):
    global parsing_table
    for i in items:
        isterminal = not i[2].startswith('`')
        if isterminal: # Shift

            cell = parsing_table.loc[(i[0]), ('action', i[2])]
            if cell !='_':
                print('conflict: '+ cell + '    s'+str(i[1]))
                continue
            parsing_table.loc[(i[0]), ('action', i[2])] = 's'+str(i[1])
        else: # goto
            parsing_table.loc[(i[0]), ('goto', i[2])] = i[1]
    # reduce
    n = Rule._n # grammar rules start index
    reduce = [(s.rules[0].lhs, s._i, Rule.augmented.index(s.rules[0].copy())) for s in states if s.hasreduce]
    for r in reduce:
        
        if r[0].endswith("'"):
            parsing_table.loc[(r[1]), ('action', '$')] = 'accept'
        else:
            for f in follow_pos(r[0]):
                cell = parsing_table.loc[(r[1]), ('action', f)]
                if cell !='_':
                    print('conflict: '+cell + '    r'+str(r[2]+n))
                parsing_table.loc[(r[1]), ('action', f)] = 'r'+str(r[2]+n)
def moves(s):
    global my_glo_result  # Global variable for storing the final result

    # Initial stack and input
    snap = []
    stack = [('$', State._n)]
    input_ = s.split() + ['$']
    action = []
    node_stack = []

    while True:
        # Accessing top of the stack and current input
        state_id = stack[-1][1]
        current_input = input_[0]

        # Retrieve action from the parsing table
        a = parsing_table.loc[(state_id), ('action', current_input)]
        action.append(a)
        snap.append((''.join([f"{s[0]}{s[1]}" for s in stack]), ' '.join(input_)))

        # Handle accept state
        if a == 'accept':
            if len(node_stack) == 1:
                # Evaluate the final node in the stack and store the result
                my_glo_result = node_stack[0].evaluate()
                print('Expression result:', my_glo_result)
                break
            else:
                raise ValueError(f"Unexpected number of nodes in node stack at accept state: {len(node_stack)}")

        # Handle shift actions
        elif a.startswith('s'):
            state_id = int(a[1:])
            stack.append((current_input, state_id))
            # Add nodes to the stack as needed
            if current_input == 'N':
                # Ensure the value is added correctly to the node stack
                value = None
                k = 0
                for a_s in all_scope:
                    if type(a_s) is not int:
                        k+=1
                    else:
                        value = a_s
                        break
                    
                if all_scope[0] == '-' or all_scope[0] == '+' or all_scope[0] == '*' or all_scope[0] == '/':
                    pass #value = 0
                else:
                    if all_scope:
                        all_scope.pop(0)
                    #value = int(all_scope.pop(0)) if all_scope else None
                node_stack.append(LiteralNode(value))
            input_.pop(0)

        # Handle reduce actions
        elif a.startswith('r'):
            rule_index = int(a[1:])
            rule = Rule.augmented[rule_index]
            num_symbols_rhs = len(rule.rhs)

            # Check if there are enough elements in node stack for reduction
            if len(node_stack)+5 < num_symbols_rhs:
                raise SyntaxError(f"Insufficient elements in node stack for reduction: rule {rule}")

            # Perform the reduction
            reduced_nodes = node_stack[-num_symbols_rhs:]
            global back
            if back is None:
                back = reduced_nodes[0].value
            del node_stack[-num_symbols_rhs:]

            # Create a new AST node for the reduction and append it to the node stack
            if rule.lhs == '`E':
                if rule.rhs == ['`E', '+', '`T']:
                    #print('E->E+T')
                    left = reduced_nodes[0]
                    right = reduced_nodes[1]
                    node_stack.append(BinaryOperationNode(left, right, '+'))
                elif rule.rhs == ['`E', '-', '`T']:
                    #print('E->E-T')
                    left = None
                    right = None
                    left = reduced_nodes[0]
                    if len(reduced_nodes) >= 2:
                        right = reduced_nodes[1]
                    node_stack.append(BinaryOperationNode(left, right, '-'))
                else:
                    pass
                    #print('E->T')
            elif rule.lhs == '`T':
                if rule.rhs == ['`T', '*', 'N']:
                    #print('T->T*N')
                    left = reduced_nodes[0]
                    right = reduced_nodes[1]
                    node_stack.append(BinaryOperationNode(left, right, '*'))
                elif rule.rhs == ['`T', '/', 'N']:
                    #print('T->T/N')
                    left = reduced_nodes[0]
                    right = reduced_nodes[1]
                    node_stack.append(BinaryOperationNode(left, right, '/'))
                else:
                    #print('T->N')
                    node_stack.append(reduced_nodes[0])
            elif rule.lhs == 'N':
                # For integer literals, just add the node back
                node_stack.append(reduced_nodes[0])

            # Update the stack for the current reduction
            for _ in range(num_symbols_rhs):
                stack.pop()

            # Retrieve the goto state and update the stack
            goto_state_id = parsing_table.loc[(stack[-1][1]), ('goto', rule.lhs)]
            stack.append((rule.lhs, goto_state_id))

        # Handle invalid parsing table entries
        else:
            raise SyntaxError(f"Invalid parsing table entry '{a}' at state {state_id} with input '{current_input}'")

    # Return the DataFrame containing the parser actions and stack snapshots
    return df(data=list(zip([s[0] for s in snap], [s[1] for s in snap], action)), columns=('Stack', 'Input', 'Action'))

def run(grammar):
    global parsing_table
    start_state, symbols = augment(grammar)
    start_state.closure()
    states.append(start_state)
    items = goto_operation()
    parsing_table =  parsing_table_skelton(symbols[0], symbols[1])
    slr_parsing_table(items)
    return items


def test(grammar, test_string):
    states_graph = run(grammar)
    for s in states:
        print('', end='') #print(s, end='\n')

   # draw(states_graph)

    print(parsing_table)

    driver_table = moves(test_string) # lexemes of test_string  must separated with spaces
    print(driver_table)






class RecursiveDescentParser:
    def __init__(self, lexemes, tokens):
        # Initialize the parser with the lexemes and tokens
        self.lexemes = lexemes
        self.tokens = tokens
        self.index = 0
        self.current_lexeme = self.lexemes[self.index]
        self.current_token = self.tokens[self.index]

    def advance(self):
        # Advance to the next token and lexeme
        self.index += 1
        if self.index < len(self.tokens):
            self.current_lexeme = self.lexemes[self.index]
            self.current_token = self.tokens[self.index]
        else:
            self.current_lexeme = None
            self.current_token = None

    def parse(self):
        # Start the parsing process
        print("Start!!")
        print("enter E")
        result = self.parse_E()
        print("exit E")
        return result

    def parse_E(self):
        # Parse rule: E -> T E'
        result = self.parse_T()
        print("enter E'")
        result = self.parse_EPrime(result)
        print("exit E'")
        return result

    def parse_EPrime(self, left_result):
        # Parse rule: E' -> + T E' | epsilon
        while self.current_token == '+':
            #print("Matched +")
            self.advance()
            right_result = self.parse_T()
            left_result += right_result
        print("epsilon")
        return left_result

    def parse_T(self):
        # Parse rule: T -> N T'
        print("enter T")
        result = self.parse_N()
        print("enter T'")
        result = self.parse_TPrime(result)
        print("exit T'")
        print("exit T")
        return result

    def parse_TPrime(self, left_result):
        # Parse rule: T' -> * N T' | epsilon
        while self.current_token == '*':
            #print("Matched *")
            self.advance()
            right_result = self.parse_N()
            left_result *= right_result
        print("epsilon")
        return left_result

    def parse_N(self):
        # Parse rule: N -> id (number)
        #print("enter N")
        if self.current_token == 'N':
            # Convert the lexeme to an integer to perform arithmetic operations
            result = int(self.current_lexeme)
            #print(f"Matched id: {result}")
            self.advance()
        else:
            raise SyntaxError("Expected 'N' (number)")
        #print("exit N")
        return result


#
#
#
#
#




if __name__ == '__main__':
    my_grammar = """
    `E => `E + `T 
    `E => `E - `T 
    `E => `T 
    `T => `T * N 
    `T => `T / N 
    `T => N"""

    S = SyntaxAnalyzer()
    expressionBottom_up_p = "5-12*12"
    #expression = "7+5*3" # <- working, test status: success
    lexemes, tokens = S.lexer(expressionBottom_up_p)
    #global all_scope
    all_scope = lexemes
    print("Lexemes:" + str(lexemes))
    print("Tokens:" + str(tokens))
    apart = ' '.join(tokens[:-1])
    test(my_grammar, apart)

    print('----------------------')
    expressionTop_down_p = "100*12"
    lexemes, tokens = S.lexer(expressionTop_down_p)
    myParser = RecursiveDescentParser(lexemes, tokens)
    myTestingResult = myParser.parse()
    print(f"myResult: {myTestingResult}")








