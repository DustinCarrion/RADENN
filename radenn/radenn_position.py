class Position:
    def __init__(self, idx, line, col, file_name, text):
        self.idx = idx
        self.line = line
        self.col = col
        self.file_name = file_name
        self.text = text
        
    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1
        if current_char == "\n":
            self.line += 1
            self.col = 0
        return self
    
    def copy(self):
        return Position(self.idx, self.line, self.col, self.file_name, self.text)
    
    