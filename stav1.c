#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>


#define TAPE_SIZE 2048
#define MACHINE_COUNT 50
#define RAID 4
#define STEPS 10000000
#define HEBBIAN_LEARNING_RATE 0.05
#define HEBBIAN_THRESHOLD 0.001
#define DOPAMINE_GAIN 10.0
#define SWITCH_STEP 100000

double *Global_Tape;
uint64_t state;
uint64_t xorshift()
{
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

uint16_t wrap_index(int index)
{
    return ((index % TAPE_SIZE) + TAPE_SIZE) % TAPE_SIZE;
}

typedef struct ProgramTreeNode
{
    char state_label;
    double w_left, w_center, w_right, bias;
    int move_left, move_right;
    struct ProgramTreeNode *next_left, *next_right;
    struct ProgramTreeNode *parent;
    int target_index;
    double prediction_error;
} ProgramTreeNode;

typedef struct
{
    int id;
    ProgramTreeNode *root;
    ProgramTreeNode *current;
    double dt;
    double kernel[2 * RAID + 1];
    double last_prediction[TAPE_SIZE];
} Machine;

void *create_shared_tape()
{
    int fd = open("shared_tape.bin", O_RDWR | O_CREAT, 0666);
    ftruncate(fd, sizeof(double) * TAPE_SIZE);
    void *addr = mmap(NULL, sizeof(double) * TAPE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    return addr;
}

void ZeroTape(double *tape)
{
    for (int i = 0; i < TAPE_SIZE; i++)
        tape[i] = 0.0;
}

double RandomizeTape(double *tape)
{
    for (int i = 0; i < TAPE_SIZE; i++)
        tape[i] = ((double)(xorshift() & 0xFFFF) / 65535.0) * 2.0 - 1.0;
}

double GaussianTarget(int i)
{
    double center = TAPE_SIZE / 2.0;
    double sigma = TAPE_SIZE / 6.0;
    double x = (double)i;
    double g = exp(-0.5 * pow((x - center) / sigma, 2.0));

    return g * 2.0 - 1.0; // normalize to [-1, 1]
}

double LinearRamp(int i)
{
    return 2.0 * ((double)i / (TAPE_SIZE - 1)) - 1.0;
}

double SineWave(int i)
{
    return sin(2.0 * M_PI * i / TAPE_SIZE);
}

double LocalDecisionRule(int i)
{
    int li = wrap_index(i - 1), ri = wrap_index(i + 1);
    double avg = (Global_Tape[li] + Global_Tape[ri]) / 2.0;

    return (Global_Tape[i] > avg) ? 1.0 : -1.0;
}

double DoubleGaussian(int i)
{
    double center1 = TAPE_SIZE / 3.0;
    double center2 = 2.0 * TAPE_SIZE / 3.0;
    double sigma = TAPE_SIZE / 12.0;
    double x = (double)i;
    double g1 = exp(-0.5 * pow((x - center1) / sigma, 2.0));
    double g2 = exp(-0.5 * pow((x - center2) / sigma, 2.0));

    return (g1 + g2) * 2.0 - 1.0;
}

double PiecewiseTarget(int i)
{
    if (i < TAPE_SIZE / 3)
        return sin(2 * M_PI * i / (TAPE_SIZE / 3));
    else if (i < 2 * TAPE_SIZE / 3)
        return (2.0 * i / TAPE_SIZE) - 1.0; // Linear ramp
    else
        return exp(-0.5 * pow((i - (TAPE_SIZE * 5 / 6.0)) / (TAPE_SIZE / 20.0), 2.0)) * 2.0 - 1.0;
}

double StepWithNoise(int i)
{
    double base = (i < TAPE_SIZE / 2) ? -1.0 : 1.0;
    return base + 0.1 * ((xorshift() % 200) / 100.0 - 1.0); // noise in [-0.1, 0.1]
}

double SymbolicCopyInvert(int i)
{
    if (i < TAPE_SIZE / 2)
        return ((xorshift() % 2000) / 1000.0) - 1.0; // [-1, 1] random

    else
        return -SymbolicCopyInvert(i - TAPE_SIZE / 2);
}

double DynamicTarget(int i, int step)
{
    if ((step / 3000) % 2 == 0)
        return DoubleGaussian(i);
    else if (step / 6000 % 2 == 0){
        return PiecewiseTarget(i);
    }else if (step / 9000 % 2 == 0){
        return StepWithNoise(i);
    } else if (step / 12000 % 2 == 0) {
        return LinearRamp(i);
    } else {
        return ((xorshift() & 0xFFFF) / 65535.0) * 2.0 - 1.0;
    }
  
}

double BayesianUpdate(double prior, double likelihood, double confidence)
{
    return (1.0 - confidence) * prior + confidence * likelihood;
}

double ComputeEntropy(double *tape)
{
    int bins = 16, hist[16] = {0};
    for (int i = 0; i < TAPE_SIZE; i++)
    {
        int b = (int)((tape[i] + 1.0) / 2.0 * bins);
        if (b < 0)
            b = 0;
        if (b >= bins)
            b = bins - 1;
        hist[b]++;
    }
    double entropy = 0.0;
    for (int i = 0; i < bins; i++)
    {
        if (hist[i] > 0)
        {
            double p = (double)hist[i] / TAPE_SIZE;
            entropy -= p * log2(p);
        }
    }

    return entropy;
}

double ComputeMSE(double *tape)
{
    double mse = 0.0;
    for (int i = 0; i < TAPE_SIZE; i++)
    {
        double t = GaussianTarget(i); // Example target function
        double d = t - tape[i];
        mse += d * d;
    }

    return mse / TAPE_SIZE;
}

ProgramTreeNode *NewNode(char label)
{
    ProgramTreeNode *node = malloc(sizeof(ProgramTreeNode));
    node->state_label = label;
    node->w_left = ((double)(xorshift() & 0xFFFF) / 65535.0) * 4.0 - 2.0;
    node->w_center = ((double)(xorshift() & 0xFFFF) / 65535.0) * 4.0 - 2.0;
    node->w_right = ((double)(xorshift() & 0xFFFF) / 65535.0) * 4.0 - 2.0;
    node->bias = ((double)(xorshift() & 0xFFFF) / 65535.0) * 2.0 - 1.0;
    node->move_left = (xorshift() & 1) ? -1 : 1;
    node->move_right = (xorshift() & 1) ? -1 : 1;
    node->target_index = xorshift() % TAPE_SIZE;
    node->next_left = node->next_right = node->parent = NULL;
    node->prediction_error = 0.0;

    return node;
}

ProgramTreeNode *BuildTreeHelper(int depth, ProgramTreeNode *parent)
{
    char label = 'A' + (xorshift() % 26);
    ProgramTreeNode *node = NewNode(label);
    node->parent = parent;
    if (depth <= 0)
    {
        if (parent)
        {
            node->next_left = parent;

            node->next_right = parent;
        }
        else
        {
            node->next_left = node;
            node->next_right = node;
        }
    }
    else
    {
        node->next_left = BuildTreeHelper(depth - 1, node);
        node->next_right = BuildTreeHelper(depth - 1, node);
    }

    return node;
}

ProgramTreeNode *BuildRandomProgramTree(int depth)
{
    return BuildTreeHelper(depth, NULL);
}

void InitMachineKernelRandom(Machine *mach)
{
    const int size = 2 * RAID + 1;
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        mach->kernel[i] = ((double)(xorshift() & 0xFFFF)) / 65535.0;
        sum += mach->kernel[i];
    }
    for (int i = 0; i < size; i++)
        mach->kernel[i] /= sum;
}

void PropagateErrorUp(ProgramTreeNode *node, double err, double decay)
{
    if (!node)
        return;
    node->prediction_error += err;
    if (node->parent)
        PropagateErrorUp(node->parent, err * decay, decay);
}

void RunMachines(Machine *machines, int machine_count, int steps)
{
    FILE *log = fopen("tape_log.csv", "w");
    FILE *elog = fopen("entropy_log.csv", "w");
    fprintf(elog, "step,entropy,mse");
    /* Compute initial MSE for reward calculation */
    double last_mse = ComputeMSE(Global_Tape);
    double mse = last_mse;
    for (int step = 0; step < steps; step++)
    {
        /* Log current tape values */
        for (int i = 0; i < TAPE_SIZE; i++)
        {
            fprintf(log, "%.5f", Global_Tape[i]);
            if (i < TAPE_SIZE - 1)
            {
                fprintf(log, ",");
            }
            else
            {
                fprintf(log, "");
            }
        }
        /* Reset prediction_error accumulators */
        for (int m = 0; m < machine_count; m++)
        {
            ProgramTreeNode *n = machines[m].current;
            while (n)
            {
                n->prediction_error = 0.0;
                n = n->parent;
            }
        }
        /* Parallel machine updates */

#pragma omp parallel for
        for (int m = 0; m < machine_count; m++)
        {
            Machine *mach = &machines[m];
            if (mach->current == NULL)
            {
                continue;
            }
            int i = mach->current->target_index;
            /* Compute posterior prediction */
            double posterior = 0.0;
            for (int k = -RAID; k <= RAID; k++)
            {
                int idx = wrap_index(i + k);
                int li = wrap_index(idx - 1);
                int ri = wrap_index(idx + 1);
                double input = mach->current->w_left * Global_Tape[li] + mach->current->w_center * Global_Tape[idx] + mach->current->w_right * Global_Tape[ri] + mach->current->bias;
                double est = tanh(input);
                posterior += mach->kernel[k + RAID] * est;
            }
            /* Clamp posterior to [-1, 1] */
            if (posterior > 1.0)
                posterior = 1.0;
            if (posterior < -1.0)
                posterior = -1.0;
            /* Select target based on phase */
            double target;
            // if (step < SWITCH_STEP)
            //{
            target = DynamicTarget(i, step);
            //}
            /*else if (step == SWITCH_STEP)
            {
                ZeroTape(Global_Tape); // once-only reset
                target = posterior;
            }
            else
            {
                target = posterior;
            }*/
            /* Compute Bayesian update */
            double conf = 0.5 + 0.5 * fabs(posterior);
            double before = Global_Tape[i];
            double after = BayesianUpdate(before, target, conf);
            double delta = after - before;
            /* Accumulate error for credit assignment */

#pragma omp critical
            PropagateErrorUp(mach->current->parent, target - posterior, 0.7);
            /* Hebbian + credit-assignment weight updates during training */
            if (step < SWITCH_STEP)
            {
                double reward = last_mse - mse;
                double dopamine = tanh(DOPAMINE_GAIN * reward);
                double lr = dopamine * HEBBIAN_LEARNING_RATE;
                /* Update weights along the node path */
                ProgramTreeNode *n = mach->current;
                while (n)
                {
                    int idxn = n->target_index;
                    int lin = wrap_index(idxn - 1);
                    int rin = wrap_index(idxn + 1);
                    double err_n = n->prediction_error;
                    if (fabs(err_n) > HEBBIAN_THRESHOLD)
                    {
                        n->w_left += lr * err_n * Global_Tape[lin];
                        n->w_center += lr * err_n * Global_Tape[idxn];
                        n->w_right += lr * err_n * Global_Tape[rin];
                        n->bias += lr * err_n;
                    }
                    n->prediction_error = 0.0;
                    n = n->parent;
                }
            }
            /* Apply tape update */

#pragma omp atomic
            Global_Tape[i] += after - before;
            /* Advance machine state */
            ProgramTreeNode *next;
            int move;
            if (after < 0)
            {
                next = mach->current->next_left;
                move = mach->current->move_left;
            }
            else
            {
                next = mach->current->next_right;
                move = mach->current->move_right;
            }
            if (next != NULL)
            {
                mach->current = next;
                mach->current->target_index = wrap_index(i + move);
            }
            else
            {
                mach->current = NULL;
            }
        }
        /* Compute MSE and entropy after all updates */
        mse = ComputeMSE(Global_Tape);
        double entropy = ComputeEntropy(Global_Tape);
        fprintf(elog, "%d,%.6f,%.6f", step, entropy, mse);
        last_mse = mse;
    }
    fclose(log);
    fclose(elog);
}

int main()
{
    Global_Tape = (double *)create_shared_tape();
    state = time(NULL);
    // state =123456789; // Fixed seed for reproducibility
    // ZeroTape(Global_Tape);
    RandomizeTape(Global_Tape);
    Machine machines[MACHINE_COUNT];
    for (int i = 0; i < MACHINE_COUNT; i++)
    {
        machines[i].id = i;
        machines[i].root = BuildRandomProgramTree(20);
        machines[i].current = machines[i].root;
        InitMachineKernelRandom(&machines[i]);
        if (machines[i].current)
            machines[i].current->target_index = (i * TAPE_SIZE) / MACHINE_COUNT;
    }
    RunMachines(machines, MACHINE_COUNT, STEPS);

    return 0;
}