import onnx 
import argparse


# print(onnx.helper.printable_graph(onnx_model.graph))
# print('Model :\n\n{}'.format(onnx.helper.printable_graph(model.graph)))
# print(onnx_model.graph)

def print_tensor_data(initializer: onnx.TensorProto) -> None:

    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        print(initializer.float_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT32:
        print(initializer.int32_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT64:
        print(initializer.int64_data)
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        print(initializer.double_data)
    elif initializer.data_type == onnx.TensorProto.DataType.UINT64:
        print(initializer.uint64_data)
    else:
        raise NotImplementedError

    return

def main(args):
    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)

    graph_def = model.graph

    BATCH = args.batch
    print(f'HEIGHT: {args.input_shape[0]}, WIDTH: {args.input_shape[1]}')
    HEIGHT, WIDTH = args.input_shape

    print('OUTPUT: ')
    outputs = graph_def.output
    for graph_output in outputs:
        output_shape = []
        # # just print shape of output node
        # for d in graph_output.type.tensor_type.shape.dim:
        #     print(d.dim_value)
        #     if d.dim_value == 0:
        #         output_shape.append(None)
        #     else:
        #         output_shape.append(d.dim_value)
        # print(f"Output Name: {graph_output.name}, Output Data Type: {graph_output.type.tensor_type.elem_type}, Output Shape: {output_shape}")

        # # modify output
        d = graph_output.type.tensor_type.shape.dim
        d[0].dim_value=BATCH  # .dim_value for number, .dim_param for ?
        # d[2].dim_param='?'
        # d[3].dim_param='?'
        print(d)
        # break

    print('\nINPUT: ')
    inputs = graph_def.input
    for graph_input in inputs:
        input_shape = []
        d = graph_input.type.tensor_type.shape.dim
        d[0].dim_value=BATCH
        d[2].dim_value=HEIGHT      # height
        d[3].dim_value=WIDTH       # width
        print(d)

    # onnx.save(model, f'batch_{BATCH}_static_input_output_scrfd_person.onnx')
    onnx.save(model, f'{args.save_onnx_dir}/batch_{BATCH}.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='modify onnx')
    # general
    parser.add_argument('--onnx', default='vit_t_e40_dynamic_sim.onnx', help='path to onnx (do not use static onnx with onnxsim)')
    parser.add_argument('--batch', default=1, type=int, help='1 or 5 or 10')
    parser.add_argument('--input-shape', nargs='+', type=int, help='input like this: --input-shape H W')
    parser.add_argument('--save-onnx-dir', default='modified_onnx', type=str, help='directory to save onnx')
    main(parser.parse_args())


