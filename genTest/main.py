# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import random


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


class op:
    name = ""
    type = ""
    size = []
    input = []
    output = ""
    def __init__(self, name, type,size, input):
        self.name = name
        self.type = type
        self.size = size
        self.input = input
    
# Press the green button in the gutter to run the script.

def get_result_size(op_type,size1,size2):
    if op_type == "Add" or op_type == "Mul":
        if size1 == size2:
            return size1
        else: 
            return None
    elif op_type == "MatMul":
        if size1[0] == size2[0] and size1[2] == size2[1]:
            return [size1[0], size1[1], size2[2]]
        else:
            return None
        

def get_op_depth(op):
    max = 0
    for i in op.input:
        d = get_op_depth(i)
        if d > max:
            max = d
    return max + 1

def get_all_op(op, ops):
    for i in op.input:  
        ops.add(i)
        get_all_op(i, ops)

def print_instr(op):
    if op.type == "Const":
        return "%" + op.name + " = " + " \"tosa.const\"() {value = dense<" + str(random.randint(0,10)) + ".0>: tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>} : () -> tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>"
    elif op.type == "Add":
        return "%" + op.name + " = " + " \"tosa.add\"(%" + op.input[0].name + ", %" + op.input[1].name + ") : (tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>, tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>) -> tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>"
    elif op.type == "Mul":
        return "%" + op.name + " = " + " \"tosa.mul\"(%" + op.input[0].name + ", %" + op.input[1].name + ") {shift = 0 : i32} : (tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>, tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>) -> tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>"
    elif op.type == "MatMul":
        return "%" + op.name + " = " + " \"tosa.matmul\"(%" + op.input[0].name + ", %" + op.input[1].name + ") : (tensor<" + str(op.input[0].size[0]) + "x" + str(op.input[0].size[1]) + "x" + str(op.input[0].size[2]) + "xf32>, tensor<" + str(op.input[1].size[0]) + "x" + str(op.input[1].size[1]) + "x" + str(op.input[1].size[2]) + "xf32>) -> tensor<" + str(op.input[0].size[0]) + "x" + str(op.input[0].size[1]) + "x" + str(op.input[1].size[2]) + "xf32>"
    
def get_ordered_op(op, ordered_ops):
    for i in op.input:
        get_ordered_op(i, ordered_ops)
    ordered_ops.append(op)

def print_starting(i,size):
    return "module {\n\tfunc.func @test_random_"+str(i)+"() ->tensor<" + str(size[0]) + "x" + str(size[1]) + "x" + str(size[2]) + "xf32> {"

def print_indent(i):
    return "\t" * i

def print_return(op):
    return "return %" + op.name + " : tensor<" + str(op.size[0]) + "x" + str(op.size[1]) + "x" + str(op.size[2]) + "xf32>\n\t}\n}\n"

def get_cost(op):
    if op.type == "Const":
        return 0
    elif op.type == "Add":
        return op.size[0]*op.size[1]*op.size[2]
    elif op.type == "Mul":
        return op.size[0]*op.size[1]*op.size[2] * 2
    elif op.type == "MatMul":
        return op.input[0].size[0] * op.input[0].size[1] * op.input[0].size[2] * op.input[1].size[2] * 3

if __name__ == '__main__':
    N_list = [1, 2, 4]
    H_list = [4, 8, 16, 32, 64, 128, 256]
    C_list = [4, 8, 16, 32, 64, 128, 256]
    W_list = [4, 8, 16, 32, 64, 128, 256]
    ops = ["Add", "Mul", "MatMul"]
    with open('data.txt','w') as data:
        for j in range(5):
            print("Generate test" + str(j))
            inputs = set()

            N = N_list[random.randint(0, len(N_list) - 1)]
            for i in range(10):
                H = H_list[random.randint(0, len(H_list) - 1)]
                C = C_list[random.randint(0, len(C_list) - 1)]
                inputs.add(op("input" + str(i), "Const",[N,H,C], []))
            i = 0
            while i < 10:
                if len(inputs) == 0:
                    break
                l = random.choice(list(inputs))
                inputs.remove(l)
                if len(inputs) == 0:
                    inputs.add(l)
                    break
                    
                r = random.choice(list(inputs))
                inputs.remove(r)
                op_type = ops[random.randint(0, len(ops) - 1)]
                if get_result_size(op_type,l.size,r.size):
                    size = get_result_size(op_type,l.size,r.size)
                    inputs.add(op(op_type + str(i), op_type, size, [l,r]))
                    i += 1
                inputs.add(l)
                inputs.add(r)
                

            largest_op = None
            largest_depth = 0
            for i in inputs:
                # print(i.name, i.type, i.size, i.input, get_op_depth(i))
                if get_op_depth(i) > largest_depth:
                    largest_depth = get_op_depth(i)
                    largest_op = i
            print(largest_op.name, largest_op.type, largest_op.size, largest_op.input, largest_depth)

            ordered_op = []
            get_ordered_op(largest_op, ordered_op)
            for i in ordered_op:
                print(i.name, i.type, i.size)

            unique_ops = set()
            cost = 0
            # print(len(ordered_op))
            
            with open('randomTest' + str(j) +'.mlir', 'w' ) as file2:
                file2.write(print_starting(j,ordered_op[-1].size))
                file2.write('\n')
                for oop in ordered_op:
                    if(unique_ops.__contains__(oop)):
                        continue
                    unique_ops.add(oop)
                    file2.write(print_indent(2))
                    file2.write(print_instr(oop))
                    file2.write('\n')
                    cost += get_cost(oop)
                    # if op.type == "Const":
                    print(print_instr(oop),get_cost(oop))
            
                file2.write(print_indent(2))
                file2.write(print_return(ordered_op[-1]))
            data.write("Test "+str(j)+": init cost:"+str(cost) + '\n')
            # print(len(unique_ops))
            

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
