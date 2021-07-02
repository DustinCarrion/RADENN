from radenn_position import Position
from radenn_errors import IllegalCharError, ExpectedCharError
from string import ascii_letters

DIGITS = "0123456789"
LETTERS = ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

TOKENS = {
    "INT": "INT",
    "FLOAT": "FLOAT",
    "STR": "STR",
    "IDENTIFIER": "IDENTIFIER",
    "KEYWORD": "KEYWORD",
    "PLUS": "PLUS",
    "MINUS": "MINUS",
    "MUL": "MUL",
    "DIV": "DIV",
    "POW": "POW",
    "MOD": "MOD",
    "EQ": "EQ",
    "EE": "EE",
    "NE": "NE",
    "LT": "LT",
    "GT": "GT",
    "LTE": "LTE",
    "GTE": "GTE",
    "LPAREN": "LPAREN",
    "RPAREN": "RPAREN",
    "LSQUARE": "LSQUARE",
    "RSQUARE": "RSQUARE",
    "LROUND": "LROUND",
    "RROUND": "RROUND",
    "COLON": "COLON",
    "COMMA": "COMMA",
    "NEWLINE": "NEWLINE",
    "EOF": "EOF"
}

KEYWORDS = [
    "var",
    "and",
    "or",
    "not",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "do",
    "function",
    "return",
    "continue",
    "break",
    "dataset",
    "optimizer",
    "inputLayer",
    "hiddenLayer",
    "outputLayer",
    "network"
]

PROTECTED_NAMES = [
    "null",
    "true",
    "false",
    "print",
    "print_optimizers",
    "print_kernel_initializers",
    "print_activation_functions",
    "print_loss_functions",
    "input",
    "type",
    "int",
    "float",
    "str",
    "list",
    "append",
    "insert",
    "pop",
    "remove",
    "extend",
    "len",
    "slice",
    "get",
    "update",
    "get_data",
    "get_labels",
    "save",
    "load",
    "save_dataset",
    "load_dataset",
    "split",
    "compile",
    "train",
    "predict",
    "evaluate",
    "run"
]


class Token:
    def __init__(self, token_type, value=None, pos_start=None, pos_end=None):
        self.type = token_type
        self.value = value
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end.copy()

    def __repr__(self):
        if self.value:
            return f"{self.type}:{self.value}"
        return f"{self.type}"

    def matches(self, token_type, value):
        return self.type == token_type and self.value == value


class Lexer:
    def __init__(self, file_name, text):
        self.file_name = file_name
        self.text = text
        self.pos = Position(-1, 0, -1, file_name, text)
        self.current_char = None
        self.len = len(self.text)
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < self.len else None

    def make_tokens(self):
        tokens = []
        while self.current_char != None:
            if self.current_char in [" ", "\t"]:
                self.advance()
            elif self.current_char == "#":
                self.skip_comment()
            elif self.current_char in [";", "\n"]:
                tokens.append(Token(TOKENS["NEWLINE"], pos_start=self.pos))
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char in ["\"", "'"]:
                tok, error = self.make_string()
                if error:
                    return [], error
                tokens.append(tok)    
            elif self.current_char == "+":
                tokens.append(Token(TOKENS["PLUS"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "-":
                tokens.append(Token(TOKENS["MINUS"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "*":
                tokens.append(Token(TOKENS["MUL"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "/":
                tokens.append(Token(TOKENS["DIV"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "^":
                tokens.append(Token(TOKENS["POW"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "%":
                tokens.append(Token(TOKENS["MOD"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TOKENS["LPAREN"], pos_start=self.pos))
                self.advance()
            elif self.current_char == ")":
                tokens.append(Token(TOKENS["RPAREN"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "[":
                tokens.append(Token(TOKENS["LSQUARE"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "]":
                tokens.append(Token(TOKENS["RSQUARE"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "{":
                tokens.append(Token(TOKENS["LROUND"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "}":
                tokens.append(Token(TOKENS["RROUND"], pos_start=self.pos))
                self.advance()
            elif self.current_char == ":":
                tokens.append(Token(TOKENS["COLON"], pos_start=self.pos))
                self.advance()
            elif self.current_char == ",":
                tokens.append(Token(TOKENS["COMMA"], pos_start=self.pos))
                self.advance()
            elif self.current_char == "!":
                tok, error = self.make_not_equals()
                if error:
                    return [], error
                tokens.append(tok)
            elif self.current_char == "=":
                tokens.append(self.make_equals())
            elif self.current_char == ">":
                tokens.append(self.make_greater_than())
            elif self.current_char == "<":
                tokens.append(self.make_less_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f"'{char}'")

        tokens.append(Token(TOKENS["EOF"], pos_start=self.pos))
        return tokens, None

    def skip_comment(self):
        self.advance()
        more_text = True
        while self.current_char != "\n":
            if self.pos.idx == self.len:
                more_text = False
                break
            self.advance()
        if more_text:
            self.advance()

    def make_number(self):
        num_str = ""
        dot_count = 0
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in (DIGITS + "."):
            if self.current_char == ".":
                if dot_count == 1:
                    break
                dot_count += 1
            num_str += self.current_char
            self.advance()
        if dot_count == 0:
            return Token(TOKENS["INT"], int(num_str), pos_start=pos_start, pos_end=self.pos)
        return Token(TOKENS["FLOAT"], float(num_str), pos_start=pos_start, pos_end=self.pos)

    def make_identifier(self):
        id_str = ""
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in (LETTERS_DIGITS + "_"):
            id_str += self.current_char
            self.advance()
        tok_type = TOKENS["KEYWORD"] if id_str in KEYWORDS else TOKENS["IDENTIFIER"]
        return Token(tok_type, id_str, pos_start=pos_start, pos_end=self.pos)

    def make_string(self):
        string = ""
        quote_type = self.current_char
        pos_start = self.pos.copy()
        escape_char = False
        self.advance()
        escape_characters = {
            "n": "\n",
            "t": "\t"
        }
        while self.current_char != None and (self.current_char != quote_type or escape_char):
            if escape_char:
                string += escape_characters.get(self.current_char, self.current_char)
                escape_char = False
            else:
                if self.current_char == "\\":
                    escape_char = True
                else:
                    string += self.current_char
            self.advance()
        if self.current_char != quote_type:
            return None, ExpectedCharError(pos_start, self.pos, f"{quote_type}") 
        self.advance()
        return Token(TOKENS["STR"], string, pos_start, self.pos), None

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == "=":
            self.advance()
            return Token(TOKENS["NE"], pos_start=pos_start, pos_end=self.pos), None
        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

    def make_equals(self):
        tok_type = TOKENS["EQ"]
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == "=":
            tok_type = TOKENS["EE"]
            self.advance()
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = TOKENS["GT"]
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == "=":
            tok_type = TOKENS["GTE"]
            self.advance()
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = TOKENS["LT"]
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == "=":
            tok_type = TOKENS["LTE"]
            self.advance()
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)


def lex(file_name, text):
    lexer = Lexer(file_name, text)
    tokens, error = lexer.make_tokens()
    return tokens, error
