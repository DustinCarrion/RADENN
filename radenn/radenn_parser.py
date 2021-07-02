from radenn_lexer import TOKENS, PROTECTED_NAMES
from radenn_errors import InvalidSyntaxError
from radenn_nodes import *

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.last_registered_advance_count = 0
        self.advance_count = 0
        self.to_reverse_count = 0
        
    def register_advancement(self):
        self.last_registered_advance_count=1
        self.advance_count+=1
        
    def register(self, res):   
        self.last_registered_advance_count = res.advance_count
        self.advance_count += res.advance_count
        if res.error:
            self.error = res.error
        return res.node
    
    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)
    
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        if not self.error or self.last_registered_advance_count == 0:
            self.error = error
        return self


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()
        
    def advance(self):
        self.tok_idx += 1
        self.update_current_tok()
        return self.current_tok
    
    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_current_tok()
        return self.current_tok
    
    def update_current_tok(self):
        if self.tok_idx >=0 and self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
    
    def parse(self):
        res = self.statements()
        if not(res.error) and self.current_tok.type != TOKENS["EOF"]:
            res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Token cannot appear after previous tokens"))
        return res
    
    def bin_op(self, left_func, ops, right_func=None):
        if right_func == None:
            right_func = left_func
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        left = res.register(left_func())
        if res.error:
            return res
        while (self.current_tok.type in ops) or ((self.current_tok.type, self.current_tok.value) in ops):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(right_func())
            if res.error:
                return res
            left = BinOpNode(left, op_tok, right, pos_start, self.current_tok.pos_end)
        return res.success(left)
                                  
    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()
        self.skip_newline(res)
        statement = res.register(self.statement())
        if res.error:
            return res
        statements.append(statement)
        more_statements = True
        while True:
            if self.current_tok.type == TOKENS["RROUND"]:
                break
            self.skip_newline(res)
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                break
            statements.append(statement)
        
        errors = False
        newline_token = 0
        if len(statements) > 1:
            for token in self.tokens:
                if token.type == TOKENS["NEWLINE"]:
                    newline_token += 1
            if newline_token < len(statements) - 1:
                errors = True
        if errors:
            return res.failure(InvalidSyntaxError(self.tokens[0].pos_start, self.tokens[-1].pos_end, f"There are {len(statements)} statements and only {newline_token+1} breaklines"))
        return res.success(ListNode(statements, pos_start, self.current_tok.pos_end))
    
    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        
        if self.current_tok.matches(TOKENS["KEYWORD"], "return"):
            res.register_advancement()
            self.advance()
            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_end))
        
        elif self.current_tok.matches(TOKENS["KEYWORD"], "continue"):
            res.register_advancement()
            self.advance()
            return res.success(ContinueNode(pos_start, self.current_tok.pos_end))
        
        elif self.current_tok.matches(TOKENS["KEYWORD"], "break"):
            res.register_advancement()
            self.advance()
            return res.success(BreakNode(pos_start, self.current_tok.pos_end))
        
        expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'var', 'not', +, -, int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
        return res.success(expr)
                   
    def expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if self.current_tok.matches(TOKENS["KEYWORD"], "var"):
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TOKENS["IDENTIFIER"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "The identifier cannot be a RADENN keyword"))
            if self.current_tok.value in PROTECTED_NAMES:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "The identifier cannot be a RADENN reserved name"))
            var_name = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TOKENS["EQ"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '='"))
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            return res.success(VarAssignNode(var_name, expr, pos_start, self.current_tok.pos_end))
                
        node = res.register(self.bin_op(self.comp_expr, [(TOKENS["KEYWORD"], "and"),(TOKENS["KEYWORD"], "or")]))
        if res.error:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'not', +, -, int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
        return res.success(node)
    
    def comp_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if self.current_tok.matches(TOKENS["KEYWORD"], "not"):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            node = res.register(self.comp_expr())
            if res.error:
                return res
            return res.success(UnaryOpNode(op_tok, node, pos_start, self.current_tok.pos_end))
        
        node = res.register(self.bin_op(self.arith_expr, [TOKENS["EE"], TOKENS["NE"], TOKENS["LT"], TOKENS["GT"], TOKENS["LTE"], TOKENS["GTE"]]))
        if res.error:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected +, -, int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
        return res.success(node)
    
    def arith_expr(self):
        return self.bin_op(self.term, [TOKENS["PLUS"], TOKENS["MINUS"]])

    def term(self):
        return self.bin_op(self.factor, [TOKENS["MUL"], TOKENS["DIV"], TOKENS["MOD"]])
    
    def factor(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        tok = self.current_tok
        if tok.type in (TOKENS["PLUS"], TOKENS["MINUS"]):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor, pos_start, self.current_tok.pos_end))
            
        return self.power()
    
    def power(self):
        return self.bin_op(self.call, [TOKENS["POW"]], self.factor)
    
    def call(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        atom = res.register(self.atom())
        if res.error:
            return res
        if self.current_tok.type == TOKENS["LPAREN"]:
            res.register_advancement()
            self.advance()
            arg_nodes = []
            
            if self.current_tok.type == TOKENS["RPAREN"]:
                res.register_advancement()
                self.advance()
                return res.success(CallNode(atom, arg_nodes, atom.pos_start, atom.pos_end))
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')', 'not', +, -, int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
                while self.current_tok.type == TOKENS["COMMA"]:
                    res.register_advancement()
                    self.advance()
                    arg_nodes.append(res.register(self.expr()))
                    if res.error:
                        return res
                if self.current_tok.type != TOKENS["RPAREN"]:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ',' or ')'"))
                res.register_advancement()
                self.advance()   
                return res.success(CallNode(atom, arg_nodes, pos_start, self.current_tok.pos_end))
        return res.success(atom)
               
    def atom(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start
        tok = self.current_tok
        if tok.type in (TOKENS["INT"], TOKENS["FLOAT"]):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok, pos_start, self.current_tok.pos_end))
        
        elif tok.type == TOKENS["STR"]:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok, pos_start, self.current_tok.pos_end))
        
        elif tok.type == TOKENS["IDENTIFIER"]:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok, pos_start, self.current_tok.pos_end))
        
        elif tok.type == TOKENS["LPAREN"]:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TOKENS["RPAREN"]:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        
        elif tok.type == TOKENS["LSQUARE"]:
            list_expr = res.register(self.list_expr())
            if res.error:
                return res
            return res.success(list_expr)
        
        elif tok.type == TOKENS["LROUND"]:
            mat_expr = res.register(self.mat_expr())
            if res.error:
                return res
            return res.success(mat_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "dataset"):
            dataset_expr = res.register(self.dataset_expr())
            if res.error:
                return res
            return res.success(dataset_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "optimizer"):
            optimizer_expr = res.register(self.optimizer_expr())
            if res.error:
                return res
            return res.success(optimizer_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "inputLayer"):
            input_layer_expr = res.register(self.input_layer_expr())
            if res.error:
                return res
            return res.success(input_layer_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "hiddenLayer"):
            hidden_layer_expr = res.register(self.hidden_layer_expr())
            if res.error:
                return res
            return res.success(hidden_layer_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "outputLayer"):
            output_layer_expr = res.register(self.output_layer_expr())
            if res.error:
                return res
            return res.success(output_layer_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "network"):
            network_expr = res.register(self.network_expr())
            if res.error:
                return res
            return res.success(network_expr)
          
        elif tok.matches(TOKENS["KEYWORD"], "if"):
            if_expr = res.register(self.if_expr())
            if res.error:
                return res
            return res.success(if_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "for"):
            for_expr = res.register(self.for_expr())
            if res.error:
                return res
            return res.success(for_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "while"):
            while_expr = res.register(self.while_expr())
            if res.error:
                return res
            return res.success(while_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "do"):
            do_while_expr = res.register(self.do_while_expr())
            if res.error:
                return res
            return res.success(do_while_expr)
        
        elif tok.matches(TOKENS["KEYWORD"], "function"):
            func_def = res.register(self.func_def())
            if res.error:
                return res
            return res.success(func_def)
        
        return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
    
    def list_expr(self):
        res = ParseResult()
        elements = []
        pos_start = self.current_tok.pos_start.copy()
        if self.current_tok.type != TOKENS["LSQUARE"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '['"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type == TOKENS["RSQUARE"]:
            res.register_advancement()
            self.advance()
        else:
            elements.append(res.register(self.expr()))
            if res.error:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ']', 'not', +, -, int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
            while self.current_tok.type == TOKENS["COMMA"]:
                res.register_advancement()
                self.advance()
                elements.append(res.register(self.expr()))
                if res.error:
                    return res
            if self.current_tok.type != TOKENS["RSQUARE"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ',' or ']'"))
            res.register_advancement()
            self.advance()
        return res.success(ListNode(elements, pos_start, self.current_tok.pos_end))
    
    def mat_expr(self):
        res = ParseResult()
        rows = []
        pos_start = self.current_tok.pos_start.copy()
        if self.current_tok.type != TOKENS["LROUND"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type == TOKENS["RROUND"]:
            res.register_advancement()
            self.advance()
        else:
            row = res.register(self.mat_row())
            if res.error:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{' or '}'"))
            rows.append(row)
            columns = len(row)
            while self.current_tok.type == TOKENS["COMMA"]:
                res.register_advancement()
                self.advance()
                row = res.register(self.mat_row())
                if res.error:
                    return res
                if len(row) != columns:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, f"All rows must have {columns} columns"))
                rows.append(row)
            if self.current_tok.type != TOKENS["RROUND"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ',' or '}'"))
            res.register_advancement()
            self.advance()
        return res.success(MatrixNode(rows, pos_start, self.current_tok.pos_end))
    
    def mat_row(self):
        res = ParseResult()
        elements = []
        if self.current_tok.type != TOKENS["LROUND"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"))
        res.register_advancement()
        self.advance()
        elements.append(res.register(self.expr()))
        if res.error:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}', 'not', +, -, int, float, str, identifier, '(', '[', '{', 'dataset', 'optimizer', 'inputLayer', 'hiddenLayer', 'outputLayer', 'network', 'if', 'for', 'while', 'do' or 'function'"))
        while self.current_tok.type == TOKENS["COMMA"]:
            res.register_advancement()
            self.advance()
            elements.append(res.register(self.expr()))
            if res.error:
                return res
        if self.current_tok.type != TOKENS["RROUND"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ',' or '}'"))
        res.register_advancement()
        self.advance()
        return res.success(elements)
    
    def dataset_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "dataset"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'dataset'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        data = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        labels = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        return res.success(DatasetNode(data, labels, pos_start, self.current_tok.pos_end))
    
    def optimizer_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "optimizer"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'optimizer'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        type_ = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        learning_rate = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        return res.success(OptimizerNode(type_, learning_rate, pos_start, self.current_tok.pos_end))
    
    def input_layer_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "inputLayer"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'inputLayer'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        
        input_neurons = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        hidden_neurons = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        kernel_initializer = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        batch_normalization = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        dropout_percentage = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        activation_function = res.register(self.expr())
        if res.error:
            return res
        
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        return res.success(InputLayerNode(input_neurons, hidden_neurons, kernel_initializer, batch_normalization, dropout_percentage, activation_function, pos_start, self.current_tok.pos_end))
        
    def hidden_layer_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "hiddenLayer"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'hiddenLayer'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        
        neurons = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        kernel_initializer = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        batch_normalization = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        dropout_percentage = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        activation_function = res.register(self.expr())
        if res.error:
            return res
        
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        return res.success(HiddenLayerNode(neurons, kernel_initializer, batch_normalization, dropout_percentage, activation_function, pos_start, self.current_tok.pos_end))
        
    def output_layer_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "outputLayer"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'outputLayer'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        
        neurons = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        kernel_initializer = res.register(self.expr())
        if res.error:
            return res
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
         
        activation_function = res.register(self.expr())
        if res.error:
            return res
        
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        return res.success(OutputLayerNode(neurons, kernel_initializer, activation_function, pos_start, self.current_tok.pos_end))
    
    def network_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "network"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'network'"))
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        
        input_layer = res.register(self.expr())
        if res.error:
            return res
        
        hidden_layers = []
        
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        
        while self.current_tok.type == TOKENS["COMMA"]:
            res.register_advancement()
            self.advance()
            layer = res.register(self.expr())
            if res.error:
                return res
            hidden_layers.append(layer)            
            
        output_layer = hidden_layers.pop(-1)
        
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        return res.success(NetworkNode(input_layer, hidden_layers, output_layer, pos_start, self.current_tok.pos_end))
        
    def if_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        all_cases = res.register(self.if_expr_cases("if"))
        if res.error:
            return res
        cases, else_case = all_cases
        return res.success(IfNode(cases, else_case, pos_start, self.current_tok.pos_end))
    
    def elif_expr(self):
        return self.if_expr_cases("elif")
    
    def else_expr(self):
        res = ParseResult()
        else_case = None
            
        if self.current_tok.matches(TOKENS["KEYWORD"], "else"):
            res.register_advancement()
            self.advance()
        
            self.skip_newline(res)
            
            if self.current_tok.type == TOKENS["LROUND"]:
                res.register_advancement()
                self.advance()                             
                statements = res.register(self.statements())
                if res.error:
                    return res
                else_case = statements
                
                if self.current_tok.type == TOKENS["RROUND"]:
                    res.register_advancement()
                    self.advance()
                else:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"))
               
            else:
                expr = res.register(self.statement())
                if res.error:
                    return res
                else_case = expr
        
        return res.success(else_case)
    
    def elif_or_else_expr(self):
        res = ParseResult()
        cases, else_case = [], None
        
        if self.current_tok.matches(TOKENS["KEYWORD"], "elif"):
            all_cases = res.register(self.elif_expr())
            if res.error:
                return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.else_expr())
            if res.error:
                return res
        
        return res.success([cases, else_case])
    
    def if_expr_cases(self, case_keyword):
        res = ParseResult()
        cases = []
        else_case = None
        if not self.current_tok.matches(TOKENS["KEYWORD"], case_keyword):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, f"Expected '{case_keyword}'"))
        res.register_advancement()
        self.advance()
        
        condition = res.register(self.expr())
        if res.error:
            return res
        
        self.skip_newline(res)
                
        if self.current_tok.type == TOKENS["LROUND"]:
            res.register_advancement()
            self.advance()
            statemets = res.register(self.statements())
            if res.error:
                return res
            cases.append([condition, statemets])
            if self.current_tok.type != TOKENS["RROUND"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"))
            res.register_advancement()
            self.advance()
            self.skip_newline(res)
            if self.current_tok.matches(TOKENS["KEYWORD"], "elif") or self.current_tok.matches(TOKENS["KEYWORD"], "else"):
                all_cases = res.register(self.elif_or_else_expr())
                if res.error:
                    return res
                new_cases, else_case = all_cases
                cases.extend(new_cases)
            
        else:
            expr = res.register(self.statement())
            if res.error:
                return res
            cases.append([condition, expr])
            self.skip_newline(res)
            all_cases = res.register(self.elif_or_else_expr())
            if res.error:
                return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)
        return res.success([cases, else_case])
                
    def for_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "for"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'for'"))
        res.register_advancement()
        self.advance()
        
        if self.current_tok.type != TOKENS["LPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register_advancement()
        self.advance()
        
        if self.current_tok.type != TOKENS["IDENTIFIER"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected identifier"))
        var_name = self.current_tok
        res.register_advancement()
        self.advance()
        
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        start_value = res.register(self.expr())
        if res.error:
            return res
        
        if self.current_tok.type != TOKENS["COMMA"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ','"))
        res.register_advancement()
        self.advance()
        
        end_value = res.register(self.expr())
        if res.error:
            return res
        
        if self.current_tok.type == TOKENS["COMMA"]:
            res.register_advancement()
            self.advance()
            step_value = res.register(self.expr())
            if res.error:
                return res
        else:
            step_value = None
        
        if self.current_tok.type != TOKENS["RPAREN"]:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register_advancement()
        self.advance()
        
        self.skip_newline(res)
        
        if self.current_tok.type == TOKENS["LROUND"]:
            res.register_advancement()
            self.advance()
            body = res.register(self.statements())
            if res.error:
                return res
            if self.current_tok.type != TOKENS["RROUND"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"))
            res.register_advancement()
            self.advance()
            return res.success(ForNode(var_name, start_value, end_value, step_value, body, pos_start, self.current_tok.pos_end))
        
        body = res.register(self.statement())
        if res.error:
            return res
        
        return res.success(ForNode(var_name, start_value, end_value, step_value, body, pos_start, self.current_tok.pos_end))
    
    def while_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "while"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'while'"))
        res.register_advancement()
        self.advance()
        
        condition = res.register(self.expr())
        if res.error:
            return res
        
        self.skip_newline(res)
        
        if self.current_tok.type == TOKENS["LROUND"]:
            res.register_advancement()
            self.advance()
            body = res.register(self.statements())
            if res.error:
                return res
            if self.current_tok.type != TOKENS["RROUND"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"))
            res.register_advancement()
            self.advance()
            return res.success(WhileNode(condition, body, pos_start, self.current_tok.pos_end))
        
        body = res.register(self.statement())
        if res.error:
            return res
        
        return res.success(WhileNode(condition, body, pos_start, self.current_tok.pos_end))
    
    def do_while_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "do"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'do'"))
        res.register_advancement()
        self.advance()
    
        self.skip_newline(res)
        
        if self.current_tok.type == TOKENS["LROUND"]:
            res.register_advancement()
            self.advance()
            body = res.register(self.statements())
            if res.error:
                return res
            if self.current_tok.type != TOKENS["RROUND"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"))
            res.register_advancement()
            self.advance()
        else:
            body = res.register(self.statement())
            if res.error:
                return res
            self.skip_newline(res)
        
        if not self.current_tok.matches(TOKENS["KEYWORD"], "while"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'while'"))
        res.register_advancement()
        self.advance()
        
        condition = res.register(self.expr())
        if res.error:
            return res
        
        return res.success(DoWhileNode(body, condition, pos_start, self.current_tok.pos_end))
    
    def func_def(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        if not self.current_tok.matches(TOKENS["KEYWORD"], "function"):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'function'"))
        res.register_advancement()
        self.advance()
        
        if self.current_tok.type == TOKENS["IDENTIFIER"]:
            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TOKENS["LPAREN"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        else:
            var_name_tok = None
            if self.current_tok.type != TOKENS["LPAREN"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected identifier or '('"))
        res.register_advancement()
        self.advance()
        
        arg_name_toks = []
        if self.current_tok.type == TOKENS["IDENTIFIER"]:
            arg_name_toks.append(self.current_tok)
            res.register_advancement()
            self.advance()
            
            while self.current_tok.type == TOKENS["COMMA"]:
                res.register_advancement()
                self.advance()
                if self.current_tok.type != TOKENS["IDENTIFIER"]:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected identifier"))
                arg_name_toks.append(self.current_tok)
                res.register_advancement()
                self.advance()
            
            if self.current_tok.type != TOKENS["RPAREN"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ',' or ')'"))
        else:
            if self.current_tok.type != TOKENS["RPAREN"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected identifier or ')'"))
        res.register_advancement()
        self.advance()
        
        self.skip_newline(res)
        
        if self.current_tok.type == TOKENS["LROUND"]:
            res.register_advancement()
            self.advance()
            body = res.register(self.statements())
            if res.error:
                return res
            if self.current_tok.type != TOKENS["RROUND"]:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"))
            res.register_advancement()
            self.advance()
            return res.success(FuncDefNode(var_name_tok, arg_name_toks, body, False, pos_start, self.current_tok.pos_end))
        
        body = res.register(self.statement())
        if res.error:
            return res
        return res.success(FuncDefNode(var_name_tok, arg_name_toks, body, True, pos_start, self.current_tok.pos_end))
    
    def skip_newline(self, res):
        while self.current_tok.type == TOKENS["NEWLINE"]:
            res.register_advancement()
            self.advance()
            
            
def parse(tokens):
    parser = Parser(tokens)
    ast = parser.parse()
    return ast.node, ast.error
