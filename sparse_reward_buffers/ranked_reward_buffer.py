



import chex
import jax
import jax.numpy as jnp
from sparse_reward_buffers.sparse_buffer import ReplayBufferState, SparseReplayBuffer, SparseRewardExperience


class R2ReplayBuffer(SparseReplayBuffer):
    def __init__(self,
        quantile: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.quantile = quantile

    
    def sample(self, key: jax.random.PRNGKey, state: ReplayBufferState) -> chex.ArrayTree:
        chex.assert_shape(state.buffer_items.reward, (self.batch_size, self.capacity,))

        # account for unpopulated rewards
        # dont just take the quantile naively!
        mask = jnp.logical_and(state.has_reward, state.populated)
        rewards = jnp.where(
            mask,
            state.buffer_items.reward,
            state.buffer_items.reward.max()
        )
        adj_quantile = self.quantile * (mask.sum() / (self.capacity * self.batch_size))

        quantile_value = jnp.quantile(rewards, adj_quantile)
        key, subkey = jax.random.split(key)
        rand_bools = jax.random.bernoulli(subkey, 0.5, state.buffer_items.reward.shape)

        def rank_rewards(reward, bools):
            return jnp.where(
                reward < quantile_value, 
                -1,
                jnp.where(
                    reward > quantile_value,
                    1,
                    jnp.where(
                        bools,
                        1,
                        -1
                    )
                )
            )
        
        ranked_rewards = rank_rewards(state.buffer_items.reward, rand_bools)
        return super().sample(key, state.replace(buffer_items=SparseRewardExperience(
            reward = ranked_rewards,
            experience = state.buffer_items.experience
        )))
