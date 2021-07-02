class NumberNode:
    def __init__(self, tok, pos_start, pos_end):
        self.tok = tok
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"{self.tok}"


class StringNode:
    def __init__(self, tok, pos_start, pos_end):
        self.tok = tok
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"{self.tok}"


class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end


class MatrixNode:
    def __init__(self, row_nodes, pos_start, pos_end):
        self.row_nodes = row_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end


class DatasetNode:
    def __init__(self, data_node, labels_node, pos_start, pos_end):
        self.data_node = data_node
        self.labels_node = labels_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class OptimizerNode:
    def __init__(self, type_node, learning_rate_node, pos_start, pos_end):
        self.type_node = type_node
        self.learning_rate_node = learning_rate_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class InputLayerNode:
    def __init__(self, input_neurons_node, hidden_neurons_node, kernel_initializer_node, batch_normalization_node, dropout_node, activation_function_node, pos_start, pos_end):
        self.input_neurons_node = input_neurons_node
        self.hidden_neurons_node = hidden_neurons_node
        self.kernel_initializer_node = kernel_initializer_node
        self.batch_normalization_node = batch_normalization_node
        self.dropout_node = dropout_node
        self.activation_function_node = activation_function_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class HiddenLayerNode:
    def __init__(self, neurons_node, kernel_initializer_node, batch_normalization_node, dropout_node, activation_function_node, pos_start, pos_end):
        self.neurons_node = neurons_node
        self.kernel_initializer_node = kernel_initializer_node
        self.batch_normalization_node = batch_normalization_node
        self.dropout_node = dropout_node
        self.activation_function_node = activation_function_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class OutputLayerNode:
    def __init__(self, neurons_node, kernel_initializer_node, activation_function_node, pos_start, pos_end):
        self.neurons_node = neurons_node
        self.kernel_initializer_node = kernel_initializer_node
        self.activation_function_node = activation_function_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class NetworkNode:
    def __init__(self, input_layer_node, hidden_layers_node, output_layer_node, pos_start, pos_end):
        self.input_layer_node = input_layer_node
        self.hidden_layers_node = hidden_layers_node
        self.output_layer_node = output_layer_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class VarAccessNode:
    def __init__(self, var_name_tok, pos_start, pos_end):
        self.var_name_tok = var_name_tok
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"{self.var_name_tok}"


class VarAssignNode:
    def __init__(self, var_name_tok, value_node, pos_start, pos_end):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"{self.var_name_tok} = {self.value_node}"


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node, pos_start, pos_end):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"({self.left_node}, {self.op_tok}, {self.right_node})"


class UnaryOpNode:
    def __init__(self, op_tok, node, pos_start, pos_end):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"({self.op_tok}, {self.node})"


class IfNode:
    def __init__(self, cases, else_case, pos_start, pos_end):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[-1][0]).pos_end


class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, pos_start, pos_end):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class WhileNode:
    def __init__(self, condition_node, body_node, pos_start, pos_end):
        self.condition_node = condition_node
        self.body_node = body_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class DoWhileNode:
    def __init__(self, body_node, condition_node, pos_start, pos_end):
        self.body_node = body_node
        self.condition_node = condition_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return, pos_start, pos_end):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.should_auto_return = should_auto_return
        self.pos_start = pos_start
        self.pos_end = pos_end


class CallNode:
    def __init__(self, node_to_call, arg_nodes, pos_start, pos_end):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end


class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return
        self.pos_start = pos_start
        self.pos_end = pos_end


class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end
