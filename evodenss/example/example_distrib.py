import torch
import torch.multiprocessing as mp
import time
# Assuming `individuals` is your list or iterable of neural network individuals

def function(i, device):
    if device == torch.device('cuda:0'):
        print(f"Function {i} is using device {device}")
        time.sleep(10)
    else:
        print(f"Function {i} is using device {device}")
        time.sleep(3.5)
    
def main(gpu_id, task_queue):
    device = torch.device(f'cuda:{gpu_id}')
    while not task_queue.empty():
        task = task_queue.get()
        print(f"Task {task} is using device {device}")
        function(task, device)
        
if __name__ == "__main__":
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    num_gpus = torch.cuda.device_count()
    task_queue = mp.Queue()
    for i in range(10):
        task_queue.put(i)

    # Start processes with available GPUs in a round-robin way
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=main, args=(gpu_id, task_queue))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


#in pratica ho una coda condivisa, passo a entrambe le gpu la coda e man mano loro la svuotano
#un elemento alla volta
#nel mio caso potrei fare una coda condivisa con gli individui, poi a ciascuna gpu
#passo la coda