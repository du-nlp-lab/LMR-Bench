

import argparse
from utils import *
# from attack import rephase_sentence
import sys

def get_sentence(demo1, x, args_method, args_cot_trigger, args_direct_answer_trigger_for_zeroshot):
    return k

sys.stdout.reconfigure(encoding='utf-8')

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)

    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    print(args.method)

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        demo = None

    total = 0
    correct_list = []
    with open(args.output_dir, "a") as wp:

        t = 0
        for i, data in enumerate(dataloader):
            t = t + 1
            if t > 3:
                break
            if i < args.resume_id - 1:
            # if i < 297:
                continue
            output_line = {}
            
            print('*************************')
            print("{}st data".format(i+1))
                    
            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()

            
            output_line["question"] = x
            output_line["gold_ans"] = y

            demo1 = demo

            flag = True
            i = 0
            # if not correct, try to add one more step
            while(flag):
                #demo1 = rephase_sentence(demo1, args, 1)
                print("demo1", demo1)

                i=i+1
                #demo1 = rephase_sentence(demo1,1)

                def get_sentence(demo1, x, args):
                    demo1 = str(demo1)
                    if args.method == "zero_shot":
                        k = x + " " + args.direct_answer_trigger_for_zeroshot
                    elif args.method == "zero_shot_cot":
                        k = x + " " + args.cot_trigger
                    elif args.method == "few_shot":
                        k = demo1 + x
                    elif args.method == "few_shot_cot":
                        k = demo1 + x
                    elif args.method == "auto_cot":
                        k = demo1 + x + " " + args.cot_trigger
                    else:
                        raise ValueError("method is not properly defined ...")
                    return k

                k = get_sentence(demo1, x, args)
                torch.save(demo1, f"unit_test/demo1.pt")
                torch.save(x, f"unit_test/x.pt")
                torch.save(k, f"unit_test/k.pt")
                # 建议只保存可序列化的 args 字段
                torch.save(args.method, f"unit_test/args_method.pt")
                torch.save(args.direct_answer_trigger_for_zeroshot, f"unit_test/args_direct_answer_trigger_for_zeroshot.pt")
                torch.save(args.cot_trigger, f"unit_test/args_cot_trigger.pt")

                # Answer experiment by generating text ...
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                z = decoder.decode(args, k, max_length)

                output_line["rationale"] = z

                if args.method == "zero_shot_cot":
                    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                    max_length = args.max_length_direct
                    pred = decoder.decode(args, z2, max_length)
                    print(z2 + pred)
                else:
                    pred = z
                    print(x)
                    #print(x + pred)
                    print(pred)

                
                pred = answer_cleansing(args, pred)


                output_line["pred_ans"] = pred
                output_line["wrap_que"] = x
                if(i==1):
                    flag = False
                    print("it is too much")

                if (np.array([pred]) == np.array([y])):
                    flag = False
                    print("correct")
                else:
                     print("wrong")


                output_json = json.dumps(output_line)
                wp.write(output_json + '\n')

                # Choose the most frequent answer from the list ...
                print("pred : {}".format(pred))
                print("GT : " + y)
                print('*************************')


                # Checking answer ...
                correct = (np.array([pred]) == np.array([y])).sum().item()
                correct_list.append(correct)
                total += 1 #np.array([y]).size(0)

                if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                    break
                #raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="multiarith", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/multiarith_manual", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3-xl", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "code-davinci-002"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="few_shot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/multiarith", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT-3"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task100.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/json1.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip100.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters100.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args

if __name__ == "__main__":
    main()