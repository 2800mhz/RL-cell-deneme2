# environment_advanced.py - Gelişmiş RL Ortamı

import gym
from gym import spaces
import numpy as np
from cell_advanced import AdvancedCell, CellCyclePhase, MetabolicState

class AdvancedCellEnv(gym.Env):
    """
    Gelişmiş hücre simülasyonu için Gym ortamı
    - Continuous action space
    - Detaylı reward shaping
    - Curriculum learning desteği
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        # Konfigürasyon
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 2000)
        self.difficulty = self.config.get('difficulty', 1.0)  # Curriculum için
        
        # Hücre
        self.cell = AdvancedCell()
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.cell.get_state_dim(),),
            dtype=np.float32
        )
        
        # Continuous action space: 12 metabolik süreç yoğunluğu
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.cell.get_action_dim(),),
            dtype=np.float32
        )
        
        # Tracking
        self.current_step = 0
        self.episode_reward = 0
        self.prev_state = None
        self.reward_components = {}
        
        # Hedefler (curriculum)
        self.survival_target = self.config.get('survival_target', 500)
        self.growth_target = self.config.get('growth_target', 1.5)
        self.division_target = self.config.get('division_target', False)
    
    def reset(self):
        """Ortamı sıfırla"""
        self.cell = AdvancedCell()
        self.current_step = 0
        self.episode_reward = 0
        self.prev_state = self.cell.get_state()
        self.reward_components = {
            'survival': 0,
            'energy': 0,
            'growth': 0,
            'health': 0,
            'efficiency': 0,
            'division': 0
        }
        
        # Difficulty'ye göre başlangıç koşullarını zorlaştır
        if self.difficulty > 1:
            stress_factor = (self.difficulty - 1) * 0.2
            self.cell.metabolites['glucose'].amount *= (1 - stress_factor)
            self.cell.metabolites['oxygen'].amount *= (1 - stress_factor)
        
        return self.cell.get_state()
    
    def step(self, action):
        """Bir adım at"""
        self.current_step += 1
        self.prev_state = self.cell.get_state()
        
        # Aksiyonu uygula
        action_results = self.cell.execute_action(action)
        
        # Pasif süreçler
        self.cell.passive_processes()
        
        # Ödül hesapla
        reward = self._calculate_reward(action, action_results)
        self.episode_reward += reward
        
        # Bitti mi?
        done = self._check_done()
        
        # State ve info
        state = self.cell.get_state()
        info = self._get_info(action_results)
        
        return state, reward, done, info
    
    def _calculate_reward(self, action, results) -> float:
        """
        Çok boyutlu reward shaping
        """
        reward = 0.0
        
        # === 1. HAYATTA KALMA ===
        if self.cell.is_alive():
            survival_bonus = 1.0
            self.reward_components['survival'] += survival_bonus
            reward += survival_bonus
        else:
            death_penalty = -100.0
            reward += death_penalty
            return reward
        
        # === 2. ENERJİ YÖNETİMİ ===
        energy_charge = self.cell.get_energy_charge()
        
        # Optimal enerji aralığı: 0.8-0.95
        if 0.8 <= energy_charge <= 0.95:
            energy_reward = 5.0
        elif 0.6 <= energy_charge < 0.8:
            energy_reward = 2.0
        elif energy_charge < 0.5:
            energy_reward = -10.0 * (0.5 - energy_charge)
        else:
            energy_reward = 1.0
        
        self.reward_components['energy'] += energy_reward
        reward += energy_reward
        
        # === 3. BÜYÜME ===
        size_delta = self.cell.size - 1.0
        growth_reward = size_delta * 10.0
        self.reward_components['growth'] += growth_reward * 0.1
        reward += growth_reward * 0.1
        
        # Protein sentez hızı bonusu
        if self.cell.protein_synthesis_rate > 0:
            reward += self.cell.protein_synthesis_rate * 2.0
        
        # === 4. SAĞLIK ===
        total_damage = (self.cell.dna_damage + 
                       self.cell.protein_damage + 
                       self.cell.membrane_damage) / 3
        
        if total_damage < 10:
            health_reward = 3.0
        elif total_damage < 30:
            health_reward = 1.0
        else:
            health_reward = -(total_damage - 30) * 0.2
        
        self.reward_components['health'] += health_reward
        reward += health_reward
        
        # Oksidatif stres cezası
        if self.cell.oxidative_stress > 30:
            reward -= (self.cell.oxidative_stress - 30) * 0.1
        
        # Apoptoz uyarısı
        if self.cell.apoptosis_signal > 20:
            reward -= self.cell.apoptosis_signal * 0.3
        
        # === 5. VERİMLİLİK ===
        # ATP üretimi vs tüketimi
        if results.get('total_atp_change', 0) > 0:
            efficiency_reward = np.log1p(results['total_atp_change']) * 0.5
            self.reward_components['efficiency'] += efficiency_reward
            reward += efficiency_reward
        
        # Atık birikimi cezası
        waste = (self.cell.metabolites['co2'].amount + 
                self.cell.metabolites['lactate'].amount)
        if waste > 40:
            reward -= (waste - 40) * 0.05
        
        # === 6. HÜCRE DÖNGÜSÜ ===
        # Döngü ilerlemesi bonusu
        if self.cell.cell_cycle_phase != CellCyclePhase.G0:
            cycle_bonus = self.cell.cycle_progress * 0.01
            reward += cycle_bonus
        
        # Bölünme mega bonusu
        if self.cell.can_divide():
            division_reward = 100.0
            self.reward_components['division'] += division_reward
            reward += division_reward
        
        # === 7. AKSİYON KALİTESİ ===
        # Uygun aksiyonları ödüllendir
        
        # Hipoksik durumda fermentasyon iyi
        if self.cell.metabolic_state == MetabolicState.HYPOXIC:
            if action[3] > 0.5:  # Fermentasyon
                reward += 2.0
        
        # Stresli durumda antioksidan iyi
        if self.cell.oxidative_stress > 20:
            if action[8] > 0.5:  # Antioksidan
                reward += 3.0
        
        # DNA hasarında onarım iyi
        if self.cell.dna_damage > 20:
            if action[10] > 0.5:  # DNA onarımı
                reward += 3.0
        
        # S fazında DNA replikasyonu gerekli
        if self.cell.cell_cycle_phase == CellCyclePhase.S:
            if action[7] > 0.5:
                reward += 2.0
        
        return reward
    
    def _check_done(self) -> bool:
        """Episode bitmeli mi?"""
        # Ölüm
        if not self.cell.is_alive():
            return True
        
        # Maksimum adım
        if self.current_step >= self.max_steps:
            return True
        
        # Başarılı bölünme
        if self.division_target and self.cell.can_divide():
            return True
        
        return False
    
    def _get_info(self, action_results) -> dict:
        """Debug için ek bilgi"""
        info = self.cell.get_info()
        info.update({
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'reward_components': self.reward_components.copy(),
            'action_results': action_results
        })
        return info
    
    def render(self, mode='human'):
        """Görselleştirme"""
        if mode == 'human':
            print("=" * 70)
            print(f"Step: {self.current_step}/{self.max_steps} | "
                  f"Episode Reward: {self.episode_reward:.2f}")
            print(self.cell)
            print("=" * 70)
    
    def get_action_meanings(self) -> list:
        """Aksiyon açıklamaları"""
        return [
            "Glycolysis",
            "TCA Cycle",
            "Oxidative Phosphorylation",
            "Fermentation",
            "Fatty Acid Oxidation",
            "Protein Synthesis",
            "Lipid Synthesis",
            "DNA Replication",
            "Antioxidant Defense",
            "Autophagy",
            "DNA Repair",
            "Rest"
        ]


class CurriculumCellEnv(AdvancedCellEnv):
    """
    Curriculum Learning için sarmalayıcı
    Zorluk kademeli olarak artar
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.total_episodes = 0
        self.success_rate = 0
        self.recent_results = []
    
    def reset(self):
        self.total_episodes += 1
        
        # Her 100 episodda zorluğu güncelle
        if self.total_episodes % 100 == 0:
            self._update_difficulty()
        
        return super().reset()
    
    def _update_difficulty(self):
        """Başarı oranına göre zorluğu ayarla"""
        if len(self.recent_results) > 0:
            success_rate = sum(self.recent_results[-100:]) / len(self.recent_results[-100:])
            
            if success_rate > 0.7:
                self.difficulty = min(3.0, self.difficulty + 0.1)
                print(f"📈 Difficulty increased to {self.difficulty:.1f}")
            elif success_rate < 0.3:
                self.difficulty = max(1.0, self.difficulty - 0.1)
                print(f"📉 Difficulty decreased to {self.difficulty:.1f}")
    
    def step(self, action):
        state, reward, done, info = super().step(action)
        
        if done:
            # Başarı = 500+ adım hayatta kalma veya bölünme
            success = self.current_step >= 500 or self.cell.can_divide()
            self.recent_results.append(1 if success else 0)
        
        return state, reward, done, info