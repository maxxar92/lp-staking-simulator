import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np

# Enable wide mode
st.set_page_config(layout="wide")




# Constants
SIMULATION_DAYS = st.sidebar.selectbox("Simulation Days", [365, 730])
TOTAL_SUPPLY = 50000000  # 50M tokens


# User stake structure
class UserStake:
    def __init__(self, lp_amount, lockup_interval_index, started_at, estimated_apr=0):
        self.lp_amount = lp_amount
        self.lockup_interval_index = lockup_interval_index
        self.started_at = started_at
        self.estimated_apr = estimated_apr

def calculate_weight(lp_amount, lockup_multiplier):
    return lp_amount * lockup_multiplier


def calculate_total_weight_per_day(user_stakes, simulation_days, lockup_intervals, lockup_multipliers):
    total_weight_per_day = [0] * simulation_days
    for day in range(simulation_days):
        current_timestamp = day * 86400
        for stake in user_stakes:
            if stake.started_at <= current_timestamp < stake.started_at + lockup_intervals[stake.lockup_interval_index] * 86400:
                weight = calculate_weight(stake.lp_amount, lockup_multipliers[stake.lockup_interval_index])
                total_weight_per_day[day] += weight
    return total_weight_per_day

def calculate_rewards_percentage_over_time(user_stakes, simulation_days, num_users, lockup_intervals, lockup_multipliers, total_rewards):
    rewards_percentage_over_time = [[0] * simulation_days for _ in range(num_users)]
    total_weight_per_day = calculate_total_weight_per_day(user_stakes, simulation_days, lockup_intervals, lockup_multipliers)
    reward_rate_per_day = total_rewards / simulation_days
    total_tokens_staked_perday = [0] * simulation_days
    
    for day in range(simulation_days):
        current_timestamp = day * 86400
        total_rewards_for_day = 0
        user_rewards_for_day = [0] * num_users
        
        for i, stake in enumerate(user_stakes):
            if stake.started_at <= current_timestamp < stake.started_at + lockup_intervals[stake.lockup_interval_index] * 86400:
                user_weight = calculate_weight(stake.lp_amount, lockup_multipliers[stake.lockup_interval_index])
                user_rewards_for_day[i] = (reward_rate_per_day * user_weight) / total_weight_per_day[day] if total_weight_per_day[day] > 0 else 0
                total_rewards_for_day += user_rewards_for_day[i]
                total_tokens_staked_perday[day] += stake.lp_amount

        for i in range(num_users):
            rewards_percentage_over_time[i][day] = (user_rewards_for_day[i] / total_rewards_for_day) * 100 if total_rewards_for_day > 0 else 0

    return rewards_percentage_over_time, total_tokens_staked_perday

def calculate_total_rewards_per_user(rewards_percentage_over_time, simulation_days, num_users, total_rewards):
    total_rewards_per_user = [0] * num_users
    reward_rate_per_day = total_rewards / simulation_days
    for i in range(num_users):
        for day in range(simulation_days):
            total_rewards_per_user[i] += (rewards_percentage_over_time[i][day] / 100) * reward_rate_per_day
    return total_rewards_per_user

# Function to calculate the effective Annual Percentage Rate (APR) for each user
def calculate_effective_apr(user_stake, total_reward, _):
    initial_investment = user_stake.lp_amount
    lockup_duration_in_years = lockup_intervals[user_stake.lockup_interval_index] / 365
    return (total_reward / initial_investment) / lockup_duration_in_years * 100  # APR as a percentage


# Streamlit App
st.title("Staking Simulator")

# Parameters
lockup_intervals = [30, 60, 90, 180, 360]  # Days
st.sidebar.subheader("Parameters")
st.sidebar.write("Lockup Days:", str(lockup_intervals))
lockup_multipliers = st.sidebar.text_input("Corresponding Multipliers (comma separated)", value="1, 2, 3, 6, 12")
total_rewards = st.sidebar.number_input("Total EDGE Rewards Over Simulation Period", min_value=1000, max_value=500000000, value=1000000)

# Parse lockup multipliers
try:
    lockup_multipliers = [int(multiplier.strip()) for multiplier in lockup_multipliers.split(",")]
except ValueError:
    st.sidebar.error("Invalid format for lockup multipliers. Please enter integers separated by commas.")
    st.stop()

##### APR-based Staking #####

# Option for APR-based staking
enable_apr_based_staking = st.sidebar.checkbox("Enable APR-based Staking ", value=True, help="A user only stakes if the estimated APR of his randomly chosen setting is above the APR threshold. Note that the effective APR can still be lower, as more users join the staking pool.")
num_users = None
MINIMUM_APR_THRESHOLD = None
enable_dynamic_user_selection = False
if enable_apr_based_staking:
    st.sidebar.write("APR-based parameters:")
    MINIMUM_APR_THRESHOLD = st.sidebar.slider("Minimum APR Threshold", min_value=0, max_value=100, value=25)  # Minimum APR threshold for staking in percentage
    enable_dynamic_user_selection = st.sidebar.checkbox("Enable per-day join probability", value=True, help="Users Join with a certain probability each day, instead of picking random start dates.")
    if not enable_dynamic_user_selection:
        min_users = st.sidebar.number_input("Minimum Number of Stakers", min_value=1, max_value=100, value=20)
    max_users = 10000
else:
    num_users = st.sidebar.number_input("Number of Users", min_value=1, max_value=100, value=20)

# Function to create user stakes with APR condition
def create_user_stakes_with_apr_condition(num_users, simulation_days, lockup_intervals, lockup_multipliers, total_rewards, min_apr_threshold, initial_whale=True):
    user_stakes = []
    total_weight = 0  # Total weight at the beginning
    user_count = 0
    loop_count = 0

    if initial_whale:
        lp_amount = 400000
        lockup_interval_index = len(lockup_intervals) - 1
        lockup_multiplier = lockup_multipliers[lockup_interval_index]
        started_at = 0
        user_stake = UserStake(lp_amount, lockup_interval_index, started_at, lockup_multiplier * 10) 
        user_stakes.append(user_stake)
        total_weight +=  lp_amount * lockup_multiplier  # Update total weight
        user_count += 1

    while (enable_apr_based_staking and user_count < max_users) or (not enable_apr_based_staking and user_count < num_users):
        lp_amount = sample_pareto_stake()# random.randint(50000, 400000)  # Random LP amount
        lockup_interval_index = random.randint(0, len(lockup_intervals) - 1)
        lockup_multiplier = lockup_multipliers[lockup_interval_index]

        # Calculate estimated APR for the user
        user_weight = lp_amount * lockup_multiplier
        estimated_rewards = total_rewards * (user_weight / (total_weight + user_weight))
        estimated_apr = (estimated_rewards / lp_amount) * 100

        # User stakes only if estimated APR is higher than the threshold
        if not enable_apr_based_staking or estimated_apr >= min_apr_threshold:
            start_day = random.randint(0, int(simulation_days * 0.9))
            started_at = start_day * 86400
            user_stake = UserStake(lp_amount, lockup_interval_index, started_at, estimated_apr)
            user_stakes.append(user_stake)
            total_weight += user_weight  # Update total weight
            user_count += 1
        
        loop_count += 1
        if loop_count > 100000:
            if user_count < min_users:
                st.sidebar.error("Could not find enough at least min stakers with the given APR threshold. Please adjust the threshold.")
                st.stop()
            else:
                break

    # Sort user_stakes by started_at date
    user_stakes = sorted(user_stakes, key=lambda x: x.started_at)
    
    return user_stakes, user_count


def create_user_stakes_with_dynamic_apr_condition(simulation_days, lockup_intervals, lockup_multipliers, total_rewards, min_apr_threshold, max_users, daily_probability_of_user_joining=0.1, initial_whale=True):
    user_stakes = []
    user_stakes_in_weight = []
    total_weight = 0
    user_count = 0

    if initial_whale:
        lp_amount = 400000
        lockup_interval_index = len(lockup_intervals) - 1
        lockup_multiplier = lockup_multipliers[lockup_interval_index]
        started_at = 0
        user_stake = UserStake(lp_amount, lockup_interval_index, started_at, lockup_multiplier * 10) 
        user_stakes.append(user_stake)
        user_stakes_in_weight.append(user_stake)
        total_weight +=  lp_amount * lockup_multiplier  # Update total weight
        user_count += 1

    for current_day in range(simulation_days):
        # Simulate a random probability for user decision
        if np.random.rand() < daily_probability_of_user_joining:#day_adjusted_probability: 
            lp_sum = 0
            while lp_sum < total_rewards / daily_probability_of_user_joining  / float(simulation_days):
                lp_amount = sample_pareto_stake()# random.randint(20000, 200000)
                lp_sum += lp_amount
                lockup_interval_index = random.randint(0, len(lockup_intervals) - 1)
                lockup_multiplier = lockup_multipliers[lockup_interval_index]

                # Calculate estimated APR for the user
                user_weight = lp_amount * lockup_multiplier
                estimated_rewards = total_rewards * (user_weight / (total_weight + user_weight))
                estimated_apr = (estimated_rewards / lp_amount) * 100

                # User stakes only if estimated APR is higher than the threshold
                if not enable_apr_based_staking or estimated_apr >= min_apr_threshold:
                    started_at = current_day * 86400
                    user_stake = UserStake(lp_amount, lockup_interval_index, started_at, estimated_apr)
                    user_stakes.append(user_stake)
                    user_stakes_in_weight.append(user_stake)
                    total_weight += user_weight
                    user_count += 1
                    # remove users from weight if they are done
                    if user_count >= max_users:
                        break

    return user_stakes, user_count


######## Calculate and Plot Stake Share Over Time ########

if enable_dynamic_user_selection:
    daily_probability_of_user_joining = st.sidebar.slider("Daily Probability of User Joining", min_value=0.01, max_value=1.0, value=0.85, help="Each day, a user tries to join the staking pool with this probability (only actually joins, if the estimated APR is above the threshold). Using a high value, only the highest lockup factors typically will join, as all others fall below the expected APR threshold. Using lower values, more options are present over time, but especially the beginning period might not be modelled realistically as many would join at the beginning.")



st.sidebar.subheader("User stake Distribution")
# Sidebar inputs for min and max stake amounts
min_stake_amount = st.sidebar.number_input("Minimum Stake Amount", min_value=1000, max_value=1000000, value=20000)
max_stake_amount = st.sidebar.number_input("Maximum Stake Amount", min_value=1000, max_value=1000000, value=400000)
add_initial_whale = st.sidebar.checkbox("Add Initial Whale", value=True, help="Adds a large initial staker (400k, one year locked) at the beginning of the simulation")


def sample_pareto_stake(shape=1.05):
    # Sample from a Pareto distribution and scale it to the range [min_stake, max_stake]
    stake = (np.random.pareto(shape) + 1) * min_stake_amount
    return min(max_stake_amount, stake)

if enable_dynamic_user_selection:
    user_stakes, num_users = create_user_stakes_with_dynamic_apr_condition(
        SIMULATION_DAYS, lockup_intervals, lockup_multipliers, total_rewards, MINIMUM_APR_THRESHOLD, max_users, daily_probability_of_user_joining, initial_whale=add_initial_whale
    )
else:
    # Choose staking function based on the APR-based staking option
    user_stakes, num_users = create_user_stakes_with_apr_condition(
        num_users, SIMULATION_DAYS, lockup_intervals, lockup_multipliers, total_rewards, MINIMUM_APR_THRESHOLD, initial_whale=add_initial_whale
    )


# Calculate the histogram of staked LP amounts
staked_lp_histogram = [user.lp_amount for user in user_stakes]

# Plotting the histogram of staked LP amounts
fig3, ax = plt.subplots()
plt.hist(staked_lp_histogram, bins=100, edgecolor='black')
plt.xlabel('Staked EDGE Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Staked EDGE Amounts')

# Display the histogram
st.sidebar.subheader("Histogram of Staked EDGE Amounts")
st.sidebar.pyplot(fig3)


# Calculate the percentage of rewards received by each user over time
rewards_percentage_over_time, lpstaked_by_day = calculate_rewards_percentage_over_time(
    user_stakes, 
    SIMULATION_DAYS, 
    num_users, 
    lockup_intervals, 
    lockup_multipliers, 
    total_rewards
)

# Calculate total rewards per user
total_rewards_per_user = calculate_total_rewards_per_user(
    rewards_percentage_over_time, 
    SIMULATION_DAYS, 
    num_users, 
    total_rewards
)
show_legend = st.sidebar.checkbox("Show Individual Stakes Legend", value=True)

# Plotting the stake share over time
fig1, ax = plt.subplots(figsize=(10, 10) if show_legend else (10, 5))
days = range(SIMULATION_DAYS)
plt.stackplot(days, *rewards_percentage_over_time)

# Sidebar checkbox for showing/hiding the legend

# Adding legend with total rewards, initial staked amount, staking length, and effective APR
if show_legend:
    legend_labels = [
        f'User {i + 1} (Staked: {user_stakes[i].lp_amount}, '
        f'Lockup: {lockup_intervals[user_stakes[i].lockup_interval_index]} days, '
        f'Mult: {lockup_multipliers[user_stakes[i].lockup_interval_index]}, '
        f'Rewards: {total_rewards_per_user[i]:.2f}) '
        f'Estimated APR: {user_stakes[i].estimated_apr:.2f}% '
        f'Effective APR: {calculate_effective_apr(user_stakes[i], total_rewards_per_user[i], SIMULATION_DAYS):.2f}%)'
        for i in range(num_users)
    ]
    plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Day')
plt.ylabel('Percentage of Total Daily Rewards')
plt.title('Percentage-wise reward share of each staker in the pool over time')

# Plotting the total supply being staked
fig2, ax = plt.subplots()
plt.plot(days, [float(n) / 50e6 * 100 for n in lpstaked_by_day], label="Percentage of Total Supply Staked")
# plt.hlines(TOTAL_SUPPLY, xmin=0, xmax=SIMULATION_DAYS, colors='r', linestyles='dashed', label="Total Supply")
plt.legend()
plt.xlabel('Day')
plt.ylabel('Percent')
plt.title('Total Supply Being Staked Over Time')

col1, col2 = st.columns([6, 4])
with col1:
    st.subheader("Stake Share Over Time")
    st.pyplot(fig1)
with col2:
    st.subheader("Total Supply Being Staked")
    st.pyplot(fig2) 


# Statistics section
st.subheader("Statistics")
total_stakers = len(user_stakes)
total_lp_staked = sum(staked_lp_histogram)
average_stake_size = total_lp_staked / total_stakers
average_effective_apr = sum([calculate_effective_apr(user_stakes[i], total_rewards_per_user[i], SIMULATION_DAYS) for i in range(num_users)]) / num_users
average_stake_length = sum([lockup_intervals[user_stakes[i].lockup_interval_index] for i in range(num_users)]) / num_users

st.write(f"Total Number of Stakers: {total_stakers}")
st.write(f"Total Amount Staked: {total_lp_staked} EDGE")
st.write(f"Average Stake Size: {average_stake_size} EDGE")
st.write(f"Average Effective APR: {average_effective_apr} %")
st.write(f"Average Stake Length: {average_stake_length} days")



st.markdown("""
## About the Staking Simulator
This webapp simulates staking behavior in the LP staking contract over a one-year period. The simulation dynamically illustrates how various factors like lockup intervals, multipliers, and user strategies impact the staking process and reward distribution.
The interaction between user choices and overall staking behavior creates complex dynamics. For instance, as more users stake, the reward pool is shared among more participants, potentially reducing individual returns.
            
### Key Aspects of the Simulation:

- **Lockup Intervals and Multipliers**: Users can stake tokens for predefined durations, each with a corresponding multiplier. In the simulation a lockup period/multiplier are chosen randomly from the list for each user.

- **Stake amounts** per user are randomly generated within the set range following the pareto distribution (many small fish, few whales), the histogram can be seen in the sidebar. The simulation then calculates the rewards and APR for each user based on these parameters. 
            
- **Total Rewards**: The simulation distributes a fixed amount of rewards over a year (or two), with the distribution based on the staked amount, lockup period, and the number of participating users. 

- **APR-based Staking Option**: Tries to simulate market psychology, users stake only if their estimated APR exceeds a set threshold. If disabled, the simulation uses a fixed number of users who stake randomly.
    - **Estimated APR**: Calculated at the time of staking, representing the potential return based on current conditions.
    - **Effective APR**: The actual return realized over time, which may differ from the estimated APR as more users join the pool and change the staking dynamics.
    - **Per-day Join Probability**: If enabled, users join the staking pool with a certain probability each day, instead of picking random start dates. This can be used to simulate a more realistic user behavior over time. Higher values mean that users will typically join only with the highest lockup factors, as all others fall below the expected APR threshold. Lower values mean that more options are present over time, but especially the beginning period might not be modelled realistically as many would join at the beginning.
            

This tool is designed to offer both a visualization and an analysis of staking strategies, rewards distribution, and the impact of different user behaviors on the overall staking ecosystem.
""")