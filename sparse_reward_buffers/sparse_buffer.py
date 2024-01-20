import jax
import jax.numpy as jnp
import chex


@chex.dataclass(frozen=True)
class SparseRewardExperience:
    """Base class for experience objects held in sparse reward buffers."""
    reward: chex.Array
    experience: chex.ArrayTree

@chex.dataclass(frozen=True)
class ReplayBufferState:
    next_idx: int
    episode_start_idx: int
    buffer_items: SparseRewardExperience
    populated: chex.Array
    has_reward: chex.Array


class SparseReplayBuffer:
    def __init__(self,
        capacity: int,
        batch_size: int,
        sample_size: int,
    ):
        
        self.capacity = capacity
        self.batch_size = batch_size
        self.sample_size = sample_size


    def init(self, template_experience: chex.ArrayTree, template_reward: chex.Array) -> ReplayBufferState:
        template = SparseRewardExperience(
            reward = template_reward,
            experience = template_experience
        )
        return ReplayBufferState(
            next_idx = jnp.zeros((self.batch_size,), dtype=jnp.int32),
            episode_start_idx = jnp.zeros((self.batch_size,), dtype=jnp.int32),
            buffer_items = jax.tree_util.tree_map(
                lambda x: jnp.zeros((self.batch_size, self.capacity, *x.shape), dtype=x.dtype),
                template
            ),
            populated = jnp.full((self.batch_size, self.capacity,), fill_value=False, dtype=jnp.bool_),
            has_reward = jnp.full((self.batch_size, self.capacity,), fill_value=True, dtype=jnp.bool_),
        )

    
    def sample(self, key: jax.random.PRNGKey, state: ReplayBufferState) -> chex.ArrayTree:
        masked_weights = jnp.logical_and(
            state.populated, 
            state.has_reward
        ).reshape(-1)

        indices = jax.random.choice(
            key,
            self.capacity * state.populated.shape[0],
            shape=(self.sample_size,),
            replace=False,
            p = masked_weights / masked_weights.sum()
        )

        batch_indices = indices // self.capacity
        item_indices = indices % self.capacity

        return jax.tree_util.tree_map(
            lambda x: x[batch_indices, item_indices],
            state.buffer_items
        )
    
    def add_experience(self, state: ReplayBufferState, experience: chex.ArrayTree) -> ReplayBufferState:

        return state.replace(
            buffer_items = state.buffer_items.replace(
                experience = jax.tree_util.tree_map(
                    lambda x, y: x.at[state.next_idx].set(y),
                    state.buffer_items.experience,
                    experience
                )),
            next_idx = (state.next_idx + 1) % self.capacity,
            populated = state.populated.at[state.next_idx].set(True),
            has_reward = state.has_reward.at[state.next_idx].set(False)
        )
    
    def assign_rewards(self, state: ReplayBufferState, rewards: jnp.ndarray, **kwargs) -> ReplayBufferState:
        return state.replace(
            episode_start_idx = state.next_idx,
            has_reward = jnp.full_like(state.has_reward, True),
            buffer_items = state.buffer_items.replace(
                reward = jnp.where(
                    ~(state.has_reward),
                    rewards,
                    state.buffer_items.reward
                )
            )
        )

    def truncate(self, state: ReplayBufferState) -> ReplayBufferState:
        # un-assigned trajectory indices have populated set to False
        # so their buffer contents will be overwritten (eventually)
        # and cannot be sampled
        # so there's no need to overwrite them with zeros here
        return state.replace(
            next_idx = state.episode_start_idx,
            has_reward = jnp.full_like(state.has_reward, True),
            populated = jnp.where(
                ~state.has_reward,
                False,
                state.populated 
            )
        )