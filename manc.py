"""
Maximum Active Neural Cover (MANC) selection for sensitive samples.
"""

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type = str, default = 'generate/', help='Directory of sensitive samples')
    parser.add_argument('--manc_save_dir', type = str, default = 'generate/manc/', help='Directory to save MANC images')
    parser.add_argument('--model_clean', type = str, default = 'model/VGG-face-clean.pt', help='Clean model')

    parser.add_argument('--label_file', type = str, default = 'data/names.txt', help='Labels')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use gpu')
    parser.set_defaults(gpu=False)

    # Trojaned model and trigger images only for sanity check
    parser.add_argument('--model_trojaned', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    parser.add_argument('--input_dir_trojaned', type = str, default = 'data/VGGFace-Trojaned', help='Directory of trojaned data')

    model = net.VGG16FaceNet()
    model.load_state_dict(torch.load(args.model_clean))

    model_trojaned = net.VGG16FaceNet()
    model_trojaned.load_state_dict(torch.load(args.model_trojaned))

    #for m in [model, model_trojaned]:
    #    for p in m.parameters():
    #        print(p.requires_grad)
    #        p.requires_grad = True

    if args.gpu:
        torch.cuda.empty_cache()
        model.cuda()
        model_trojaned.cuda()

    if args.sanity_check:
        # Accuracy of clean and trojaned models
        # Percentage of inputs that are predicted differently
        accuracy_model_clean, accuracy_model_trojaned, model_diff = utils.eval(
            input_dir=args.input_dir_clean,
            label_file=args.label_file,
            model=model,
            gpu=args.gpu,
            model2=model_trojaned
        )
        print(f"Clean model accuracy on clean inputs: {accuracy_model_clean}")
        print(f"Trojaned model accuracy on clean inputs: {accuracy_model_trojaned}")
        print(f"model_diff on clean inputs: {model_diff}")

        # Attack success rate of the trojaned model
        _, attack_success_rate = utils.eval(
            input_dir=args.input_dir_trojaned,
            label_file=args.label_file,
            model=model_trojaned,
            gpu=args.gpu
        )
        print(f"Trojaned model on trojaned inputs attack_success_rate: {attack_success_rate}")


if __name__ == '__main__':
    main()
