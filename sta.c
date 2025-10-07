#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// --- Simulation Parameters ---
#define TAPE_SIZE 2048         // Size of the shared data array
#define MACHINE_COUNT 30       // Number of concurrent agents
#define RAID 5                 // Radius for the local kernel read (Kernel size is 2*RAID + 1)
#define STEPS 10000        // Total number of simulation steps
#define SWITCH_STEP 5000    // Step at which the simulation switches from training to self-stabilization
#define LOG_SAMPLING_RATE 1000 // Log the tape state every N steps

// --- Learning Algorithm Parameters ---
#define HEBBIAN_LEARNING_RATE 0.05  // Learning rate for weight updates
#define HEBBIAN_THRESHOLD 0.001     // Minimum prediction error to trigger a weight update
#define DOPAMINE_GAIN 10.0          // Multiplier for the global reward signal
#define ERROR_PROPAGATION_DECAY 0.5 // Decay factor for error signals propagated up the program tree

// --- Global Shared State ---
double *Global_Tape;         // The shared data tape, mapped to a file for potential inter-process communication
uint64_t *thread_rng_states; // Array of random states, one for each OpenMP thread

uint64_t xorshift(uint64_t *state);
uint16_t wrap_index(int index);

void *create_shared_tape();
void ZeroTape(double *tape);
void RandomizeTape(double *tape, uint64_t *main_rng_state);
double ComputeMSE(double *tape);
double ComputeEntropy(double *tape);

double GaussianTarget(int i);
double DynamicTarget(int i, int step, uint64_t *rng_state);

typedef struct ProgramTreeNode
{
    char state_label;
    double w_left, w_center, w_right, bias;         // Weights for the node's computation
    int move_left, move_right;                      // Movement on the tape for the next step
    struct ProgramTreeNode *next_left, *next_right; // Pointers to next states
    struct ProgramTreeNode *parent;
    int target_index;        // Current position on the tape this machine is targeting
    double prediction_error; // Accumulated error for credit assignment
} ProgramTreeNode;

typedef struct
{
    int id;
    ProgramTreeNode *root;       
    ProgramTreeNode *current;   
    double kernel[2 * RAID + 1]; 
} Machine;

ProgramTreeNode *NewNode(char label, uint64_t *rng_state);
ProgramTreeNode *BuildRandomProgramTree(int depth, uint64_t *rng_state);
void InitMachineKernelRandom(Machine *mach, uint64_t *rng_state);

uint64_t xorshift(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

uint16_t wrap_index(int index)
{
    return ((index % TAPE_SIZE) + TAPE_SIZE) % TAPE_SIZE;
}

void *create_shared_tape()
{
    int fd = open("shared_tape.bin", O_RDWR | O_CREAT, 0666);
    if (fd == -1)
    {
        perror("Failed to open shared tape file");
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd, sizeof(double) * TAPE_SIZE) == -1)
    {
        perror("Failed to set size of shared tape file");
        close(fd);
        exit(EXIT_FAILURE);
    }
    void *addr = mmap(NULL, sizeof(double) * TAPE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED)
    {
        perror("Failed to mmap shared tape file");
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);
    return addr;
}

void ZeroTape(double *tape)
{
    for (int i = 0; i < TAPE_SIZE; i++)
        tape[i] = 0.0;
}

void RandomizeTape(double *tape, uint64_t *main_rng_state)
{
    for (int i = 0; i < TAPE_SIZE; i++)
    {
        tape[i] = ((double)(xorshift(main_rng_state) & 0xFFFF) / 65535.0) * 2.0 - 1.0;
    }
}

// --- Target Functions ---
double GaussianTarget(int i)
{
    double center = TAPE_SIZE / 2.0;
    double sigma = TAPE_SIZE / 6.0;
    double x = (double)i;
    double g = exp(-0.5 * pow((x - center) / sigma, 2.0));
    return g * 2.0 - 1.0; // Normalize to [-1, 1]
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

double DynamicTarget(int i, int step, uint64_t *rng_state)
{
    if ((step / 1500) % 2 == 0)
    {
        return DoubleGaussian(i);
    }
    else
    {
        return ((xorshift(rng_state) & 0xFFFF) / 65535.0) * 2.0 - 1.0; // Random noise
    }
}

double ComputeMSE(double *tape)
{
    double mse = 0.0;
    for (int i = 0; i < TAPE_SIZE; i++)
    {
        double target = GaussianTarget(i); 
        double diff = target - tape[i];
        mse += diff * diff;
    }
    return mse / TAPE_SIZE;
}

double ComputeEntropy(double *tape)
{
    int bins = 16;
    int hist[16] = {0};
    for (int i = 0; i < TAPE_SIZE; i++)
    {
        int b = (int)(((tape[i] + 1.0) / 2.0) * bins);
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

ProgramTreeNode *NewNode(char label, uint64_t *rng_state)
{
    ProgramTreeNode *node = malloc(sizeof(ProgramTreeNode));
    if (!node)
    {
        perror("Failed to allocate memory for new ProgramTreeNode");
        exit(EXIT_FAILURE);
    }
    node->state_label = label;
    node->w_left = ((double)(xorshift(rng_state) & 0xFFFF) / 65535.0) * 4.0 - 2.0;
    node->w_center = ((double)(xorshift(rng_state) & 0xFFFF) / 65535.0) * 4.0 - 2.0;
    node->w_right = ((double)(xorshift(rng_state) & 0xFFFF) / 65535.0) * 4.0 - 2.0;
    node->bias = ((double)(xorshift(rng_state) & 0xFFFF) / 65535.0) * 2.0 - 1.0;
    node->move_left = (xorshift(rng_state) & 1) ? -1 : 1;
    node->move_right = (xorshift(rng_state) & 1) ? -1 : 1;
    node->target_index = xorshift(rng_state) % TAPE_SIZE;
    node->next_left = node->next_right = node->parent = NULL;
    node->prediction_error = 0.0;
    return node;
}

ProgramTreeNode *BuildTreeHelper(int depth, ProgramTreeNode *parent, uint64_t *rng_state)
{
    char label = 'A' + (xorshift(rng_state) % 26);
    ProgramTreeNode *node = NewNode(label, rng_state);
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
        node->next_left = BuildTreeHelper(depth - 1, node, rng_state);
        node->next_right = BuildTreeHelper(depth - 1, node, rng_state);
    }
    return node;
}

ProgramTreeNode *BuildRandomProgramTree(int depth, uint64_t *rng_state)
{
    return BuildTreeHelper(depth, NULL, rng_state);
}

void InitMachineKernelRandom(Machine *mach, uint64_t *rng_state)
{
    const int size = 2 * RAID + 1;
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        mach->kernel[i] = ((double)(xorshift(rng_state) & 0xFFFF)) / 65535.0;
        sum += mach->kernel[i];
    }
    for (int i = 0; i < size; i++)
    {
        mach->kernel[i] /= sum;
    }
}

void PropagateErrorUp(ProgramTreeNode *node, double error, double decay)
{
    while (node != NULL)
    {
        node->prediction_error += error;
        error *= decay;
        node = node->parent;
    }
}

double BayesianUpdate(double prior, double likelihood, double confidence)
{
    return (1.0 - confidence) * prior + confidence * likelihood;
}

void RunMachines(Machine *machines, int machine_count, int steps)
{
    FILE *log = fopen("tape_log.csv", "w");
    if (!log)
    {
        perror("Failed to open tape_log.csv");
        return;
    }
    FILE *elog = fopen("entropy_log.csv", "w");
    if (!elog)
    {
        perror("Failed to open entropy_log.csv");
        fclose(log);
        return;
    }
    fprintf(elog, "step,entropy,mse\n");

    double last_mse = ComputeMSE(Global_Tape);

    for (int step = 0; step < steps; step++)
    {
        if (step % LOG_SAMPLING_RATE == 0)
        {
            for (int i = 0; i < TAPE_SIZE; i++)
            {
                fprintf(log, "%.5f%c", Global_Tape[i], (i == TAPE_SIZE - 1) ? '\n' : ',');
            }
        }

        double current_mse = ComputeMSE(Global_Tape);
        double reward = last_mse - current_mse;
        double dopamine = tanh(DOPAMINE_GAIN * reward);

        if (step == SWITCH_STEP)
        {
            printf("Step %d: Switching to self-stabilization mode. Resetting tape.\n", step);
            //ZeroTape(Global_Tape);
            uint64_t *rng0 = &thread_rng_states[0];
            for (int i = 0; i < TAPE_SIZE; i++)
            {
                double tiny = ((double)(xorshift(rng0) & 0xFFFF) / 65535.0) * 1e-3;
                Global_Tape[i] = tiny;
            }
        }

#pragma omp parallel for
        for (int m = 0; m < machine_count; m++)
        {
            Machine *mach = &machines[m];
            if (mach->current == NULL)
                continue;

            int thread_id = omp_get_thread_num();
            uint64_t *rng_state = &thread_rng_states[thread_id];
            int i = mach->current->target_index;

            double posterior = 0.0;
            for (int k = -RAID; k <= RAID; k++)
            {
                int idx = wrap_index(i + k);
                int li = wrap_index(idx - 1);
                int ri = wrap_index(idx + 1);
                double input = mach->current->w_left * Global_Tape[li] + mach->current->w_center * Global_Tape[idx] + mach->current->w_right * Global_Tape[ri] + mach->current->bias;
                double estimate = tanh(input);
                posterior += mach->kernel[k + RAID] * estimate;
            }

            posterior = fmax(-1.0, fmin(1.0, posterior));

            double target;
            if (step < SWITCH_STEP)
            {
                target = DynamicTarget(i, step, rng_state);
            }
            else
            {
                target = posterior;
            }

            double confidence = 0.5 + 0.7 * fabs(posterior);
            double before = Global_Tape[i];
            double after = BayesianUpdate(before, target, confidence);
            double delta = after - before;

            double prediction_error = target - posterior;
            PropagateErrorUp(mach->current, prediction_error, ERROR_PROPAGATION_DECAY);

            if (step < SWITCH_STEP)
            {
                ProgramTreeNode *node = mach->current;
                while (node != NULL)
                {
                    if (fabs(node->prediction_error) > HEBBIAN_THRESHOLD)
                    {
                        double effective_lr = dopamine * HEBBIAN_LEARNING_RATE;
                        int idxn = node->target_index;
                        int lin = wrap_index(idxn - 1);
                        int rin = wrap_index(idxn + 1);
                        node->w_left += effective_lr * node->prediction_error * Global_Tape[lin];
                        node->w_center += effective_lr * node->prediction_error * Global_Tape[idxn];
                        node->w_right += effective_lr * node->prediction_error * Global_Tape[rin];
                        node->bias += effective_lr * node->prediction_error;
                    }
                    node->prediction_error = 0.0;
                    node = node->parent;
                }
            }

#pragma omp atomic
            Global_Tape[i] += delta;

            ProgramTreeNode *next_node;
            int move;
            if (after < 0)
            {
                next_node = mach->current->next_left;
                move = mach->current->move_left;
            }
            else
            {
                next_node = mach->current->next_right;
                move = mach->current->move_right;
            }
            mach->current = next_node;
            if (mach->current)
            {
                mach->current->target_index = wrap_index(i + move);
            }
        }

        if (step % LOG_SAMPLING_RATE == 0)
        {
            double entropy = ComputeEntropy(Global_Tape);
            fprintf(elog, "%d,%.6f,%.6f\n", step, entropy, current_mse);
        }
        last_mse = current_mse;
    }

    fclose(log);
    fclose(elog);
}

int main()
{
    Global_Tape = (double *)create_shared_tape();

    uint64_t main_rng_state = time(NULL);
    // For reproducibility:
    // uint64_t main_rng_state = 123456789;

    int max_threads = omp_get_max_threads();
    thread_rng_states = malloc(sizeof(uint64_t) * max_threads);
    if (!thread_rng_states)
    {
        perror("Failed to allocate memory for thread RNG states");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < max_threads; i++)
    {
        thread_rng_states[i] = xorshift(&main_rng_state);
    }

    printf("Initializing tape and machines...\n");
    RandomizeTape(Global_Tape, &main_rng_state);

    Machine *machines = malloc(sizeof(Machine) * MACHINE_COUNT);
    if (!machines)
    {
        perror("Failed to allocate memory for machines");
        free(thread_rng_states);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < MACHINE_COUNT; i++)
    {
        machines[i].id = i;
        machines[i].root = BuildRandomProgramTree(15, &main_rng_state);
        machines[i].current = machines[i].root;
        InitMachineKernelRandom(&machines[i], &main_rng_state);
        if (machines[i].current)
        {
            machines[i].current->target_index = (i * TAPE_SIZE) / MACHINE_COUNT;
        }
    }

    printf("Starting simulation with %d machines for %d steps...\n", MACHINE_COUNT, STEPS);
    printf("Logging tape state every %d steps.\n", LOG_SAMPLING_RATE);
    printf("Learning phase ends at step %d.\n", SWITCH_STEP);

    RunMachines(machines, MACHINE_COUNT, STEPS);

    printf("Simulation finished.\n");

    free(machines);
    free(thread_rng_states);
    munmap(Global_Tape, sizeof(double) * TAPE_SIZE);

    return 0;
}
