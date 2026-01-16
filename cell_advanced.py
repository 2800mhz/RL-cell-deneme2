# cell_advanced.py - Gelişmiş Hücre Biyokimyası

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math

class CellCyclePhase(Enum):
    """Hücre döngüsü fazları"""
    G0 = "G0"  # Dinlenme (quiescent)
    G1 = "G1"  # Büyüme 1
    S = "S"    # DNA sentezi
    G2 = "G2"  # Büyüme 2
    M = "M"    # Mitoz (bölünme)

class MetabolicState(Enum):
    """Metabolik durumlar"""
    NORMAL = "normal"
    STRESSED = "stressed"
    STARVING = "starving"
    HYPOXIC = "hypoxic"
    APOPTOTIC = "apoptotic"

@dataclass
class Metabolite:
    """Tek bir metabolit için veri yapısı"""
    name: str
    amount: float
    max_capacity: float
    diffusion_rate: float = 0.0  # Dışarıdan gelme hızı
    decay_rate: float = 0.0      # Bozunma hızı
    toxic_threshold: float = float('inf')  # Toksisite eşiği
    
    def normalize(self) -> float:
        """0-1 arası normalize değer"""
        return np.clip(self.amount / self.max_capacity, 0, 1)
    
    def is_toxic(self) -> bool:
        return self.amount > self.toxic_threshold
    
    def update(self):
        """Pasif güncelleme (difüzyon ve bozunma)"""
        # Difüzyon (dışarıdan gelme)
        if self.diffusion_rate > 0:
            deficit = self.max_capacity - self.amount
            self.amount += deficit * self.diffusion_rate
        
        # Bozunma
        if self.decay_rate > 0:
            self.amount *= (1 - self.decay_rate)
        
        # Sınırla
        self.amount = np.clip(self.amount, 0, self.max_capacity * 1.5)

@dataclass
class Enzyme:
    """Enzim kinetiği (Michaelis-Menten)"""
    name: str
    vmax: float          # Maksimum hız
    km: float            # Michaelis sabiti
    amount: float = 1.0  # Enzim miktarı
    optimal_ph: float = 7.0
    optimal_temp: float = 37.0
    
    def activity(self, substrate: float, ph: float = 7.0, temp: float = 37.0) -> float:
        """
        Michaelis-Menten kinetiği ile enzim aktivitesi
        v = (Vmax * [S]) / (Km + [S])
        """
        # Temel Michaelis-Menten
        if substrate <= 0:
            return 0
        
        base_rate = (self.vmax * substrate) / (self.km + substrate)
        
        # pH etkisi (Gauss dağılımı)
        ph_factor = np.exp(-((ph - self.optimal_ph) ** 2) / 2)
        
        # Sıcaklık etkisi
        temp_factor = np.exp(-((temp - self.optimal_temp) ** 2) / 50)
        
        return base_rate * self.amount * ph_factor * temp_factor

@dataclass  
class SignalingPathway:
    """Hücre içi sinyal yolağı"""
    name: str
    activity: float = 0.0  # 0-1 arası aktivite
    threshold: float = 0.5  # Aktivasyon eşiği
    decay: float = 0.1      # Sinyal azalma hızı
    
    def activate(self, signal_strength: float):
        """Sinyal yolağını aktive et"""
        self.activity = np.clip(self.activity + signal_strength, 0, 1)
    
    def is_active(self) -> bool:
        return self.activity > self.threshold
    
    def update(self):
        """Sinyal zamanla azalır"""
        self.activity *= (1 - self.decay)

@dataclass
class Organelle:
    """Organel simülasyonu"""
    name: str
    count: int
    efficiency: float = 1.0
    damage: float = 0.0
    
    def get_total_capacity(self) -> float:
        return self.count * self.efficiency * (1 - self.damage)

class AdvancedCell:
    """
    Gelişmiş hücre simülasyonu
    - Gerçekçi enzim kinetiği
    - Hücre döngüsü
    - Sinyal yolakları
    - Organel bazlı hesaplamalar
    - Apoptoz mekanizması
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Hücreyi başlangıç durumuna getir"""
        
        # === TEMEL ÖZELLİKLER ===
        self.age = 0
        self.size = 1.0
        self.volume = 1.0  # pL (pikolit)
        self.mass = 1.0    # pg (pikogram)
        
        # Hücre döngüsü
        self.cell_cycle_phase = CellCyclePhase.G1
        self.cycle_progress = 0.0  # 0-100, faz içindeki ilerleme
        self.division_count = 0
        
        # Metabolik durum
        self.metabolic_state = MetabolicState.NORMAL
        
        # === METABOLİTLER ===
        self.metabolites: Dict[str, Metabolite] = {
            # Enerji
            'atp': Metabolite('ATP', 100, 200, diffusion_rate=0.0, decay_rate=0.01),
            'adp': Metabolite('ADP', 20, 100),
            'amp': Metabolite('AMP', 5, 50),
            'nadh': Metabolite('NADH', 50, 100, decay_rate=0.02),
            'fadh2': Metabolite('FADH2', 30, 80, decay_rate=0.02),
            
            # Besinler
            'glucose': Metabolite('Glucose', 80, 200, diffusion_rate=0.05),
            'pyruvate': Metabolite('Pyruvate', 10, 100),
            'acetyl_coa': Metabolite('Acetyl-CoA', 20, 100),
            'amino_acids': Metabolite('Amino Acids', 60, 150, diffusion_rate=0.03),
            'fatty_acids': Metabolite('Fatty Acids', 50, 150, diffusion_rate=0.02),
            
            # Gazlar ve su
            'oxygen': Metabolite('O2', 100, 200, diffusion_rate=0.1),
            'co2': Metabolite('CO2', 5, 100, decay_rate=0.15, toxic_threshold=80),
            'water': Metabolite('H2O', 100, 200, diffusion_rate=0.08),
            
            # Atıklar
            'lactate': Metabolite('Lactate', 5, 100, decay_rate=0.05, toxic_threshold=60),
            'urea': Metabolite('Urea', 2, 50, decay_rate=0.08, toxic_threshold=40),
            'ros': Metabolite('ROS', 5, 50, decay_rate=0.1, toxic_threshold=30),  # Reaktif oksijen
            
            # Yapısal
            'proteins': Metabolite('Proteins', 80, 200, decay_rate=0.002),
            'lipids': Metabolite('Lipids', 60, 150, decay_rate=0.001),
            'nucleotides': Metabolite('Nucleotides', 40, 100, diffusion_rate=0.01),
            
            # Sinyal molekülleri
            'calcium': Metabolite('Ca2+', 0.1, 10, decay_rate=0.2),
            'camp': Metabolite('cAMP', 1, 20, decay_rate=0.15),
        }
        
        # === ENZİMLER ===
        self.enzymes: Dict[str, Enzyme] = {
            # Glikoliz
            'hexokinase': Enzyme('Hexokinase', vmax=10, km=5),
            'pfk': Enzyme('PFK', vmax=8, km=3),  # Fosfofruktokinaz
            'pyruvate_kinase': Enzyme('Pyruvate Kinase', vmax=12, km=4),
            
            # TCA döngüsü
            'citrate_synthase': Enzyme('Citrate Synthase', vmax=15, km=6),
            'isocitrate_dh': Enzyme('Isocitrate DH', vmax=10, km=5),
            
            # Elektron taşıma
            'complex_i': Enzyme('Complex I', vmax=20, km=3),
            'complex_iv': Enzyme('Complex IV', vmax=25, km=2),  # Sitokrom c oksidaz
            'atp_synthase': Enzyme('ATP Synthase', vmax=30, km=4),
            
            # Diğer
            'ldh': Enzyme('LDH', vmax=15, km=5),  # Laktat dehidrojenaz
            'catalase': Enzyme('Catalase', vmax=50, km=10),  # ROS temizleyici
            'caspase': Enzyme('Caspase-3', vmax=5, km=8, amount=0.1),  # Apoptoz
        }
        
        # === SİNYAL YOLAKLARI ===
        self.pathways: Dict[str, SignalingPathway] = {
            'ampk': SignalingPathway('AMPK', activity=0.0),  # Enerji sensörü
            'mtor': SignalingPathway('mTOR', activity=0.5),  # Büyüme kontrolü
            'p53': SignalingPathway('p53', activity=0.1),    # DNA hasarı/apoptoz
            'hif1a': SignalingPathway('HIF-1α', activity=0.0),  # Hipoksi yanıtı
            'nfkb': SignalingPathway('NF-κB', activity=0.1),    # Stres yanıtı
            'akt': SignalingPathway('Akt', activity=0.3),       # Hayatta kalma
            'mapk': SignalingPathway('MAPK', activity=0.2),     # Proliferasyon
        }
        
        # === ORGANELLER ===
        self.organelles: Dict[str, Organelle] = {
            'mitochondria': Organelle('Mitochondria', count=1000, efficiency=1.0),
            'ribosomes': Organelle('Ribosomes', count=10000, efficiency=1.0),
            'er': Organelle('ER', count=1, efficiency=1.0),
            'golgi': Organelle('Golgi', count=1, efficiency=1.0),
            'lysosomes': Organelle('Lysosomes', count=300, efficiency=1.0),
            'peroxisomes': Organelle('Peroxisomes', count=100, efficiency=1.0),
        }
        
        # === FİZYOLOJİK PARAMETRELER ===
        self.ph = 7.2  # Sitoplazmik pH
        self.temperature = 37.0  # Celsius
        self.membrane_potential = -70  # mV
        self.osmotic_pressure = 1.0  # Relatif
        
        # === HASAR VE SAĞLIK ===
        self.dna_damage = 0.0       # 0-100
        self.protein_damage = 0.0  # 0-100
        self.membrane_damage = 0.0 # 0-100
        self.oxidative_stress = 0.0  # 0-100
        
        # === BÜYÜME ===
        self.growth_rate = 0.0
        self.protein_synthesis_rate = 0.0
        self.dna_replication_progress = 0.0
        
        # === APOPTOZ ===
        self.apoptosis_signal = 0.0  # 0-100
        self.is_apoptotic = False
        
        # === İSTATİSTİKLER ===
        self.total_atp_produced = 0
        self.total_glucose_consumed = 0
        self.actions_taken = []
    
    # ==========================================================
    # TEMEL SORGULAR
    # ==========================================================
    
    def is_alive(self) -> bool:
        """Hücre canlı mı?"""
        # Kritik metabolit eksiklikleri
        if self.metabolites['atp'].amount <= 0:
            return False
        if self.metabolites['oxygen'].amount <= 0:
            return False
        if self.metabolites['water'].amount <= 10:
            return False
        
        # Aşırı hasar
        total_damage = self.dna_damage + self.protein_damage + self.membrane_damage
        if total_damage > 250:
            return False
        
        # Apoptoz tamamlandı
        if self.apoptosis_signal >= 100:
            return False
        
        return True
    
    def can_divide(self) -> bool:
        """Hücre bölünebilir mi?"""
        return (
            self.cell_cycle_phase == CellCyclePhase.M and
            self.cycle_progress >= 90 and
            self.dna_damage < 10 and
            self.metabolites['atp'].amount > 80 and
            self.size >= 1.8
        )
    
    def get_energy_charge(self) -> float:
        """
        Adenylate Energy Charge = ([ATP] + 0.5[ADP]) / ([ATP] + [ADP] + [AMP])
        Normal: 0.8-0.95
        """
        atp = self.metabolites['atp'].amount
        adp = self.metabolites['adp'].amount
        amp = self.metabolites['amp'].amount
        
        total = atp + adp + amp
        if total == 0:
            return 0
        
        return (atp + 0.5 * adp) / total
    
    def get_redox_state(self) -> float:
        """NAD+/NADH oranı (metabolik durum göstergesi)"""
        nadh = self.metabolites['nadh'].amount
        # NAD+ = max - NADH (basitleştirilmiş)
        nad = self.metabolites['nadh'].max_capacity - nadh
        if nadh == 0:
            return float('inf')
        return nad / nadh
    
    def get_metabolic_rate(self) -> float:
        """Boyut ve duruma göre metabolik hız"""
        base_rate = self.size ** 0.75  # Allometrik
        
        # Sıcaklık etkisi (Q10 = 2)
        temp_factor = 2 ** ((self.temperature - 37) / 10)
        
        # Enerji durumu etkisi
        energy_factor = self.get_energy_charge()
        
        # Hasar etkisi
        damage_factor = 1 - (self.protein_damage / 200)
        
        return base_rate * temp_factor * energy_factor * damage_factor
    
    # ==========================================================
    # METABOLİK YOLAKLAR (Detaylı)
    # ==========================================================
    
    def glycolysis(self) -> Tuple[bool, float]:
        """
        Glikoliz: Glucose → 2 Pyruvate + 2 ATP + 2 NADH
        Detaylı enzim kinetiği ile
        """
        glucose = self.metabolites['glucose'].amount
        atp = self.metabolites['atp'].amount
        
        # Hız sınırlayıcı enzim: PFK
        rate = self.enzymes['pfk'].activity(glucose, self.ph, self.temperature)
        rate *= self.get_metabolic_rate()
        
        # ATP inhibisyonu (yüksek ATP glikolizi yavaşlatır)
        if atp > 150:
            rate *= 0.5
        
        # Minimum substrat kontrolü
        glucose_needed = rate * 0.5
        atp_invest = rate * 0.1  # Başlangıç yatırımı
        
        if glucose >= glucose_needed and atp >= atp_invest:
            self.metabolites['glucose'].amount -= glucose_needed
            self.metabolites['atp'].amount -= atp_invest
            
            # Ürünler
            pyruvate_produced = glucose_needed * 2
            atp_produced = rate * 2 + atp_invest  # Net kazanç
            nadh_produced = rate * 2
            
            self.metabolites['pyruvate'].amount += pyruvate_produced
            self.metabolites['atp'].amount += atp_produced
            self.metabolites['nadh'].amount += nadh_produced
            
            # İstatistik
            self.total_atp_produced += atp_produced
            self.total_glucose_consumed += glucose_needed
            
            return True, atp_produced
        
        return False, 0
    
    def tca_cycle(self) -> Tuple[bool, float]:
        """
        TCA Döngüsü (Krebs): Acetyl-CoA → CO2 + NADH + FADH2 + GTP
        """
        acetyl_coa = self.metabolites['acetyl_coa'].amount
        
        rate = self.enzymes['citrate_synthase'].activity(acetyl_coa, self.ph)
        rate *= self.get_metabolic_rate()
        
        # Mitokondri kapasitesi
        mito_capacity = self.organelles['mitochondria'].get_total_capacity()
        rate *= (mito_capacity / 1000)  # Normalize
        
        if acetyl_coa >= rate:
            self.metabolites['acetyl_coa'].amount -= rate
            
            # Ürünler (1 Acetyl-CoA → 3 NADH + 1 FADH2 + 1 GTP + 2 CO2)
            self.metabolites['nadh'].amount += rate * 3
            self.metabolites['fadh2'].amount += rate
            self.metabolites['atp'].amount += rate  # GTP ≈ ATP
            self.metabolites['co2'].amount += rate * 2
            
            return True, rate * 3
        
        return False, 0
    
    def oxidative_phosphorylation(self) -> Tuple[bool, float]:
        """
        Oksidatif Fosforilasyon: NADH/FADH2 + O2 → ATP + H2O
        Elektron taşıma zinciri
        """
        nadh = self.metabolites['nadh'].amount
        fadh2 = self.metabolites['fadh2'].amount
        oxygen = self.metabolites['oxygen'].amount
        
        # Complex IV (sitokrom oksidaz) hızı
        electron_carriers = nadh + fadh2 * 0.67  # FADH2 daha az verimli
        rate = self.enzymes['atp_synthase'].activity(electron_carriers, self.ph)
        
        # Oksijen gereksinimi
        o2_needed = rate * 0.5
        
        # Mitokondri verimliliği
        mito = self.organelles['mitochondria']
        efficiency = mito.efficiency * (1 - mito.damage / 100)
        
        if oxygen >= o2_needed and electron_carriers > 0:
            # NADH kullan (öncelikli)
            nadh_used = min(nadh, rate * 0.8)
            fadh2_used = min(fadh2, rate * 0.2)
            
            self.metabolites['nadh'].amount -= nadh_used
            self.metabolites['fadh2'].amount -= fadh2_used
            self.metabolites['oxygen'].amount -= o2_needed
            
            # ATP üretimi: 1 NADH → ~2.5 ATP, 1 FADH2 → ~1.5 ATP
            atp_from_nadh = nadh_used * 2.5 * efficiency
            atp_from_fadh2 = fadh2_used * 1.5 * efficiency
            total_atp = atp_from_nadh + atp_from_fadh2
            
            self.metabolites['atp'].amount += total_atp
            self.metabolites['water'].amount += o2_needed * 2
            self.metabolites['adp'].amount = max(0, self.metabolites['adp'].amount - total_atp * 0.5)
            
            # ROS üretimi (yan ürün, %1-2)
            ros_leak = total_atp * 0.015 * (1 + mito.damage / 50)
            self.metabolites['ros'].amount += ros_leak
            
            self.total_atp_produced += total_atp
            return True, total_atp
        
        return False, 0
    
    def anaerobic_fermentation(self) -> Tuple[bool, float]:
        """
        Anaerobik Fermentasyon: Pyruvate → Lactate + NAD+
        Oksijen yokken
        """
        pyruvate = self.metabolites['pyruvate'].amount
        nadh = self.metabolites['nadh'].amount
        
        rate = self.enzymes['ldh'].activity(pyruvate, self.ph)
        
        nadh_needed = rate * 0.5
        
        if pyruvate >= rate and nadh >= nadh_needed:
            self.metabolites['pyruvate'].amount -= rate
            self.metabolites['nadh'].amount -= nadh_needed
            
            self.metabolites['lactate'].amount += rate
            # NAD+ rejenerasyonu (NADH kapasitesi geri gelir)
            
            # Laktat asidozu - pH düşürür
            self.ph -= rate * 0.01
            self.ph = max(6.5, self.ph)
            
            return True, 0  # Net ATP üretimi yok
        
        return False, 0
    
    def fatty_acid_oxidation(self) -> Tuple[bool, float]:
        """
        Beta-Oksidasyon: Fatty acid → Acetyl-CoA + NADH + FADH2
        """
        fa = self.metabolites['fatty_acids'].amount
        oxygen = self.metabolites['oxygen'].amount
        atp = self.metabolites['atp'].amount
        
        # Aktivasyon için ATP gerekli
        activation_cost = 2
        
        if fa >= 1 and oxygen >= 10 and atp >= activation_cost:
            rate = min(fa * 0.3, oxygen / 20)
            rate *= self.get_metabolic_rate()
            
            self.metabolites['fatty_acids'].amount -= rate
            self.metabolites['oxygen'].amount -= rate * 10
            self.metabolites['atp'].amount -= activation_cost
            
            # 1 FA (16C) → 8 Acetyl-CoA + 7 NADH + 7 FADH2
            self.metabolites['acetyl_coa'].amount += rate * 8
            self.metabolites['nadh'].amount += rate * 7
            self.metabolites['fadh2'].amount += rate * 7
            self.metabolites['co2'].amount += rate * 2
            
            return True, rate * 100  # Yüksek enerji potansiyeli
        
        return False, 0
    
    def protein_synthesis(self) -> Tuple[bool, float]:
        """
        Protein Sentezi: Amino acids + ATP → Proteins
        Ribozom kapasitesine bağlı
        """
        aa = self.metabolites['amino_acids'].amount
        atp = self.metabolites['atp'].amount
        nucleotides = self.metabolites['nucleotides'].amount
        
        # Ribozom kapasitesi
        ribosome_capacity = self.organelles['ribosomes'].get_total_capacity()
        max_rate = ribosome_capacity / 5000
        
        # mTOR aktivasyonu protein sentezini artırır
        mtor_factor = 1 + self.pathways['mtor'].activity
        
        rate = min(max_rate * mtor_factor, aa / 5, atp / 10)
        rate *= self.get_metabolic_rate()
        
        # AMPK aktifse protein sentezi yavaşlar
        if self.pathways['ampk'].is_active():
            rate *= 0.3
        
        if aa >= rate * 5 and atp >= rate * 10:
            self.metabolites['amino_acids'].amount -= rate * 5
            self.metabolites['atp'].amount -= rate * 10
            
            self.metabolites['proteins'].amount += rate * 4
            self.protein_synthesis_rate = rate
            
            # Büyüme
            self.size += rate * 0.01
            self.mass += rate * 0.01
            
            return True, rate
        
        return False, 0
    
    def lipid_synthesis(self) -> Tuple[bool, float]:
        """
        Lipid Sentezi: Acetyl-CoA + ATP → Lipids
        """
        acetyl = self.metabolites['acetyl_coa'].amount
        atp = self.metabolites['atp'].amount
        nadh = self.metabolites['nadh'].amount
        
        rate = min(acetyl / 8, atp / 14, nadh / 7)
        rate *= self.get_metabolic_rate()
        
        if acetyl >= rate * 8 and atp >= rate * 14 and nadh >= rate * 7:
            self.metabolites['acetyl_coa'].amount -= rate * 8
            self.metabolites['atp'].amount -= rate * 14
            self.metabolites['nadh'].amount -= rate * 7
            
            self.metabolites['lipids'].amount += rate * 2
            
            # Membran büyümesi
            self.size += rate * 0.005
            
            return True, rate
        
        return False, 0
    
    def dna_replication(self) -> Tuple[bool, float]:
        """
        DNA Replikasyonu (S fazında)
        """
        if self.cell_cycle_phase != CellCyclePhase.S:
            return False, 0
        
        nucleotides = self.metabolites['nucleotides'].amount
        atp = self.metabolites['atp'].amount
        
        if nucleotides >= 5 and atp >= 20:
            rate = min(nucleotides / 10, atp / 30)
            
            self.metabolites['nucleotides'].amount -= rate * 5
            self.metabolites['atp'].amount -= rate * 20
            
            self.dna_replication_progress += rate * 5
            self.cycle_progress += rate * 2
            
            # Replikasyon stresi - hafif ROS
            self.metabolites['ros'].amount += rate * 0.5
            
            return True, rate
        
        return False, 0
    
    def antioxidant_defense(self) -> Tuple[bool, float]:
        """
        Antioksidan Savunma: ROS → H2O
        Katalaz ve glutatyon peroksidaz
        """
        ros = self.metabolites['ros'].amount
        
        if ros > 0:
            rate = self.enzymes['catalase'].activity(ros, self.ph)
            rate = min(rate, ros)
            
            self.metabolites['ros'].amount -= rate
            self.metabolites['water'].amount += rate * 0.5
            
            self.oxidative_stress = max(0, self.oxidative_stress - rate * 0.5)
            
            return True, rate
        
        return False, 0
    
    def autophagy(self) -> Tuple[bool, float]:
        """
        Otofaji: Hasarlı organelleri geri dönüştür
        """
        atp = self.metabolites['atp'].amount
        
        # AMPK aktifse otofaji artar
        if not self.pathways['ampk'].is_active():
            rate = 0.1
        else:
            rate = 0.5
        
        if atp >= 5:
            self.metabolites['atp'].amount -= 5
            
            # Hasarlı proteinleri amino aside çevir
            damaged_protein = min(self.protein_damage * 0.5, 5)
            self.metabolites['amino_acids'].amount += damaged_protein * 0.5
            self.protein_damage = max(0, self.protein_damage - damaged_protein)
            
            # Lizozom kullanımı
            self.organelles['lysosomes'].efficiency *= 0.999
            
            return True, damaged_protein
        
        return False, 0
    
    def repair_dna(self) -> Tuple[bool, float]:
        """
        DNA Onarımı
        """
        atp = self.metabolites['atp'].amount
        
        if self.dna_damage > 0 and atp >= 15:
            # p53 aktifse onarım daha verimli
            repair_efficiency = 1 + self.pathways['p53'].activity
            
            repair_amount = min(5 * repair_efficiency, self.dna_damage)
            
            self.metabolites['atp'].amount -= 15
            self.dna_damage -= repair_amount
            
            return True, repair_amount
        
        return False, 0
    
    def rest(self) -> Tuple[bool, float]:
        """
        Dinlenme - minimum aktivite
        """
        # Sadece hafif stres azalması
        self.oxidative_stress = max(0, self.oxidative_stress - 0.5)
        
        # pH dengeleme
        self.ph = self.ph * 0.95 + 7.2 * 0.05  # 7.2'ye doğru
        
        return True, 0
    
    # ==========================================================
    # SİNYAL YOLAKLARI
    # ==========================================================
    
    def update_signaling(self):
        """Sinyal yolaklarını güncelle"""
        
        # === AMPK (Enerji Sensörü) ===
        energy_charge = self.get_energy_charge()
        if energy_charge < 0.7:
            self.pathways['ampk'].activate(0.3)
        elif energy_charge > 0.9:
            self.pathways['ampk'].activity *= 0.5
        
        # === mTOR (Büyüme Kontrolü) ===
        aa_level = self.metabolites['amino_acids'].normalize()
        if aa_level > 0.5 and not self.pathways['ampk'].is_active():
            self.pathways['mtor'].activate(0.2)
        else:
            self.pathways['mtor'].activity *= 0.7
        
        # === p53 (DNA Hasarı Yanıtı) ===
        if self.dna_damage > 20:
            self.pathways['p53'].activate(0.4)
        
        # === HIF-1α (Hipoksi Yanıtı) ===
        o2_level = self.metabolites['oxygen'].normalize()
        if o2_level < 0.3:
            self.pathways['hif1a'].activate(0.5)
            self.metabolic_state = MetabolicState.HYPOXIC
        
        # === NF-κB (Stres Yanıtı) ===
        if self.oxidative_stress > 30 or self.metabolites['ros'].amount > 20:
            self.pathways['nfkb'].activate(0.3)
        
        # === Akt (Hayatta Kalma) ===
        if self.metabolites['atp'].amount > 50:
            self.pathways['akt'].activate(0.1)
        
        # Tüm yolakları güncelle (decay)
        for pathway in self.pathways.values():
            pathway.update()
    
    def check_apoptosis(self):
        """Apoptoz kontrolü"""
        
        # p53 çok aktifse ve hasar yüksekse
        if self.pathways['p53'].activity > 0.7 and self.dna_damage > 50:
            self.apoptosis_signal += 5
        
        # Aşırı ROS
        if self.metabolites['ros'].amount > 40:
            self.apoptosis_signal += 3
        
        # ATP kritik
        if self.metabolites['atp'].amount < 10:
            self.apoptosis_signal += 10
        
        # Akt hayatta kalma sinyali
        if self.pathways['akt'].is_active():
            self.apoptosis_signal -= 2
        
        self.apoptosis_signal = np.clip(self.apoptosis_signal, 0, 100)
        
        # Kaspaz aktivasyonu
        if self.apoptosis_signal > 50:
            self.enzymes['caspase'].amount += 0.1
            self.is_apoptotic = True
    
    # ==========================================================
    # HÜCRE DÖNGÜSÜ
    # ==========================================================
    
    def update_cell_cycle(self):
        """Hücre döngüsünü güncelle"""
        
        # Apoptotik hücre döngüye devam etmez
        if self.is_apoptotic:
            return
        
        # Checkpoint kontrolleri
        energy_ok = self.get_energy_charge() > 0.7
        damage_ok = self.dna_damage < 30
        size_ok = self.size >= 1.0
        
        if self.cell_cycle_phase == CellCyclePhase.G0:
            # Büyüme sinyalleri yeterliyse G1'e geç
            if self.pathways['mtor'].is_active() and energy_ok:
                self.cell_cycle_phase = CellCyclePhase.G1
                self.cycle_progress = 0
        
        elif self.cell_cycle_phase == CellCyclePhase.G1:
            # Büyüme ve enerji birikimi
            if energy_ok and damage_ok:
                self.cycle_progress += self.get_metabolic_rate() * 0.5
            
            # G1/S checkpoint
            if self.cycle_progress >= 100 and size_ok and damage_ok:
                self.cell_cycle_phase = CellCyclePhase.S
                self.cycle_progress = 0
                self.dna_replication_progress = 0
        
        elif self.cell_cycle_phase == CellCyclePhase.S:
            # DNA replikasyonu progress'e göre
            if self.dna_replication_progress >= 100:
                self.cell_cycle_phase = CellCyclePhase.G2
                self.cycle_progress = 0
        
        elif self.cell_cycle_phase == CellCyclePhase.G2:
            # G2/M checkpoint
            if energy_ok and damage_ok:
                self.cycle_progress += self.get_metabolic_rate() * 0.3
            
            if self.cycle_progress >= 100:
                self.cell_cycle_phase = CellCyclePhase.M
                self.cycle_progress = 0
        
        elif self.cell_cycle_phase == CellCyclePhase.M:
            # Mitoz
            if energy_ok:
                self.cycle_progress += 2
    
    def divide(self) -> 'AdvancedCell':
        """Hücre bölünmesi"""
        if not self.can_divide():
            return None
        
        # Yeni hücre oluştur
        daughter = AdvancedCell()
        
        # Kaynakları paylaş
        for name, metabolite in self.metabolites.items():
            half = metabolite.amount / 2
            metabolite.amount = half
            daughter.metabolites[name].amount = half
        
        # Boyutu yarıla
        self.size /= 2
        self.mass /= 2
        daughter.size = self.size
        daughter.mass = self.mass
        
        # Organelleri paylaş
        for name, org in self.organelles.items():
            half = org.count // 2
            org.count = half
            daughter.organelles[name].count = half
        
        # Döngüyü sıfırla
        self.cell_cycle_phase = CellCyclePhase.G1
        self.cycle_progress = 0
        self.division_count += 1
        
        return daughter
    
    # ==========================================================
    # PASİF SÜREÇLER
    # ==========================================================
    
    def passive_processes(self):
        """Her adımda otomatik gerçekleşen süreçler"""
        
        # Yaşlan
        self.age += 1
        
        # Metabolitleri güncelle (difüzyon, decay)
        for metabolite in self.metabolites.values():
            metabolite.update()
        
        # Bazal metabolizma - ATP tüket
        maintenance = 2.0 * self.get_metabolic_rate()
        self.metabolites['atp'].amount -= maintenance
        self.metabolites['adp'].amount += maintenance * 0.8
        
        # Bazal ROS üretimi
        self.metabolites['ros'].amount += 0.5 * self.get_metabolic_rate()
        
        # Toksisite kontrolü
        for metabolite in self.metabolites.values():
            if metabolite.is_toxic():
                self.oxidative_stress += 1
                self.protein_damage += 0.5
        
        # ROS hasarı
        ros = self.metabolites['ros'].amount
        if ros > 15:
            ros_damage = (ros - 15) * 0.1
            self.dna_damage += ros_damage * 0.3
            self.protein_damage += ros_damage * 0.5
            self.membrane_damage += ros_damage * 0.2
        
        # pH hasarı
        if self.ph < 6.8 or self.ph > 7.6:
            ph_damage = abs(self.ph - 7.2) * 2
            self.protein_damage += ph_damage
        
        # Protein degradasyonu
        self.metabolites['proteins'].amount *= 0.998
        
        # Sinyal yolaklarını güncelle
        self.update_signaling()
        
        # Apoptoz kontrolü
        self.check_apoptosis()
        
        # Hücre döngüsü
        self.update_cell_cycle()
        
        # Metabolik durum güncelle
        self.update_metabolic_state()
        
        # Hasarları sınırla
        self.dna_damage = np.clip(self.dna_damage, 0, 100)
        self.protein_damage = np.clip(self.protein_damage, 0, 100)
        self.membrane_damage = np.clip(self.membrane_damage, 0, 100)
        self.oxidative_stress = np.clip(self.oxidative_stress, 0, 100)
    
    def update_metabolic_state(self):
        """Metabolik durumu güncelle"""
        if self.is_apoptotic:
            self.metabolic_state = MetabolicState.APOPTOTIC
        elif self.metabolites['oxygen'].amount < 30:
            self.metabolic_state = MetabolicState.HYPOXIC
        elif self.metabolites['glucose'].amount < 20 and self.metabolites['atp'].amount < 30:
            self.metabolic_state = MetabolicState.STARVING
        elif self.oxidative_stress > 40:
            self.metabolic_state = MetabolicState.STRESSED
        else:
            self.metabolic_state = MetabolicState.NORMAL
    
    # ==========================================================
    # RL ARAYÜZÜ
    # ==========================================================
    
    def get_state(self) -> np.ndarray:
        """
        RL için durum vektörü - 40 boyutlu
        """
        state = []
        
        # Temel (4)
        state.extend([
            self.size,
            self.age / 1000,  # Normalize
            self.get_energy_charge(),
            self.get_metabolic_rate()
        ])
        
        # Metabolitler - normalize (12)
        key_metabolites = ['atp', 'glucose', 'oxygen', 'amino_acids', 
                          'fatty_acids', 'pyruvate', 'nadh', 'water',
                          'co2', 'lactate', 'ros', 'proteins']
        for key in key_metabolites:
            state.append(self.metabolites[key].normalize())
        
        # Sinyal yolakları (7)
        for pathway in self.pathways.values():
            state.append(pathway.activity)
        
        # Hasarlar - normalize (4)
        state.extend([
            self.dna_damage / 100,
            self.protein_damage / 100,
            self.membrane_damage / 100,
            self.oxidative_stress / 100
        ])
        
        # Hücre döngüsü (5)
        cycle_one_hot = [0] * 5
        cycle_idx = list(CellCyclePhase).index(self.cell_cycle_phase)
        cycle_one_hot[cycle_idx] = 1
        state.extend(cycle_one_hot)
        
        # Diğer (3)
        state.extend([
            self.cycle_progress / 100,
            self.apoptosis_signal / 100,
            1.0 if self.can_divide() else 0.0
        ])
        
        # pH ve membran potansiyeli (2)
        state.extend([
            (self.ph - 6.5) / 1.5,  # 6.5-8.0 arası normalize
            (self.membrane_potential + 100) / 150  # -100 ile +50 arası
        ])
        
        # Organel verimlilikleri (2)
        state.append(self.organelles['mitochondria'].get_total_capacity() / 1000)
        state.append(self.organelles['ribosomes'].get_total_capacity() / 10000)
        
        return np.array(state, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        """State boyutu"""
        return 39
    
    def get_action_dim(self) -> int:
        """Aksiyon boyutu (continuous)"""
        return 12
    
    def execute_action(self, action: np.ndarray) -> Dict[str, any]:
        """
        Continuous action space - her aksiyona ne kadar kaynak ayrılacağı
        action[0]: Glikoliz yoğunluğu (0-1)
        action[1]: TCA döngüsü (0-1)
        action[2]: Oksidatif fosforilasyon (0-1)
        action[3]: Fermentasyon (0-1)
        action[4]: Yağ oksidasyonu (0-1)
        action[5]: Protein sentezi (0-1)
        action[6]: Lipid sentezi (0-1)
        action[7]: DNA replikasyonu (0-1)
        action[8]: Antioksidan savunma (0-1)
        action[9]: Otofaji (0-1)
        action[10]: DNA onarımı (0-1)
        action[11]: Dinlenme (0-1)
        """
        action = np.clip(action, 0, 1)
        results = {}
        total_atp_change = 0
        
        # Her aksiyonu yoğunluğa göre çalıştır
        if action[0] > 0.1:
            success, atp = self.glycolysis()
            results['glycolysis'] = success
            total_atp_change += atp * action[0]
        
        if action[1] > 0.1:
            success, atp = self.tca_cycle()
            results['tca'] = success
            total_atp_change += atp * action[1]
        
        if action[2] > 0.1:
            success, atp = self.oxidative_phosphorylation()
            results['oxphos'] = success
            total_atp_change += atp * action[2]
        
        if action[3] > 0.1:
            success, _ = self.anaerobic_fermentation()
            results['fermentation'] = success
        
        if action[4] > 0.1:
            success, atp = self.fatty_acid_oxidation()
            results['fat_ox'] = success
            total_atp_change += atp * action[4]
        
        if action[5] > 0.1:
            success, rate = self.protein_synthesis()
            results['protein_synth'] = success
        
        if action[6] > 0.1:
            success, _ = self.lipid_synthesis()
            results['lipid_synth'] = success
        
        if action[7] > 0.1:
            success, _ = self.dna_replication()
            results['dna_rep'] = success
        
        if action[8] > 0.1:
            success, _ = self.antioxidant_defense()
            results['antioxidant'] = success
        
        if action[9] > 0.1:
            success, _ = self.autophagy()
            results['autophagy'] = success
        
        if action[10] > 0.1:
            success, _ = self.repair_dna()
            results['dna_repair'] = success
        
        if action[11] > 0.5:
            success, _ = self.rest()
            results['rest'] = success
        
        results['total_atp_change'] = total_atp_change
        return results
    
    def get_info(self) -> Dict:
        """Detaylı bilgi"""
        return {
            'alive': self.is_alive(),
            'age': self.age,
            'size': self.size,
            'cell_cycle': self.cell_cycle_phase.value,
            'cycle_progress': self.cycle_progress,
            'energy_charge': self.get_energy_charge(),
            'metabolic_state': self.metabolic_state.value,
            'atp': self.metabolites['atp'].amount,
            'dna_damage': self.dna_damage,
            'protein_damage': self.protein_damage,
            'oxidative_stress': self.oxidative_stress,
            'apoptosis_signal': self.apoptosis_signal,
            'can_divide': self.can_divide(),
            'division_count': self.division_count
        }
    
    def __str__(self):
        alive = "✓ ALIVE" if self.is_alive() else "✗ DEAD"
        return (
            f"AdvancedCell [{alive}]\n"
            f"  Age: {self.age} | Size: {self.size:.2f} | Phase: {self.cell_cycle_phase.value}\n"
            f"  Energy Charge: {self.get_energy_charge():.2f} | ATP: {self.metabolites['atp'].amount:.1f}\n"
            f"  State: {self.metabolic_state.value}\n"
            f"  DNA Damage: {self.dna_damage:.1f}% | Oxidative Stress: {self.oxidative_stress:.1f}%\n"
            f"  Apoptosis: {self.apoptosis_signal:.1f}%"
        )