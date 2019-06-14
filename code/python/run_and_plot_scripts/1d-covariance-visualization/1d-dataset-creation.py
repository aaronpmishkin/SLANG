import argparse
import numpy as np

arg_definitions = {
    '--seed':{
        'dest':'seed', 'type':int, 'default':1,
        'help':'[Int] Seed for the RNG'
    },
    '--N':{
        'dest':'N', 'type':int, 'default':30,
        'help':'[Int] Number of samples to generate'
    },
    '--plot':{
        'dest':'plot', 'type':bool, 'default':False,
        'help':'[True/False] Plot dataset before exiting'
    },
    '--out':{
        'dest':'outfile', 'type':str, 'default':'1d-cov-viz-dataset',
        'help':'[Int] Path to the output file'
    },
}

parser = argparse.ArgumentParser(
    description='Creates a 1D dataset for qualitative evaluation of covariances.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

for arg, arg_def in arg_definitions.items():
    parser.add_argument(arg, **arg_def)

X_RANGE = [-4, 4]
NOISE_LOC = 0
NOISE_SCALE = 3

def generate_dataset(args, with_outlier=False, with_gap=False):
    np.random.seed(args.seed)

    if with_gap:
        x1 = np.random.uniform(low=X_RANGE[0], high=X_RANGE[0]/4, size=int(args.N/2))
        x2 = np.random.uniform(low=X_RANGE[1]/4, high=X_RANGE[1], size=int(args.N/2))
        x = np.concatenate([x1, x2])
    else:
        x = np.random.uniform(low=X_RANGE[0], high=X_RANGE[1], size=args.N)

    if with_outlier:
        x[-1] = X_RANGE[1]*2

    noise = np.random.normal(loc=NOISE_LOC, scale=NOISE_SCALE, size=args.N)
    y = x**3 + noise

    return x, y

def plot_dataset(x, y):
    import matplotlib.pyplot as plt

    plt.plot(x, y, '.')

    plt.ylim([
        -max([abs(i) for i in plt.ylim()]),
        max([abs(i) for i in plt.ylim()])
    ])

    plt.grid()
    plt.show()

def save(x, y, outfile):
    np.savetxt(outfile, np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1), delimiter=",", header="x, y")

if __name__ == "__main__":
    args = parser.parse_args()

    x, y = generate_dataset(args)
    if args.plot:
        plot_dataset(x, y)
    save(x, y, args.outfile+".csv")
    
    x, y = generate_dataset(args, with_outlier=True)
    if args.plot:
        plot_dataset(x, y)
    save(x, y, args.outfile+"_outlier.csv")
    
    x, y = generate_dataset(args, with_gap=True)
    if args.plot:
        plot_dataset(x, y)
    save(x, y, args.outfile+"_gap.csv")