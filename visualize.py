# visualize.py - Basit Matplotlib ile Görselleştirme

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from environment import CellEnv
import os

class CellVisualizer:
    """
    Matplotlib ile basit hücre simülasyonu görselleştirme
    """
    def __init__(self, model_path=None, max_steps=500):
        self.env = CellEnv(max_steps=max_steps)
        self.model = None
        
        if model_path and os.path.exists(f"{model_path}.zip"):
            print(f"Model yükleniyor: {model_path}")
            self.model = PPO.load(model_path)
        else:
            print("Model bulunamadı, rastgele aksiyonlar kullanılacak")
        
        self.history = {
            'step': [],
            'size': [],
            'atp': [],
            'glucose': [],
            'oxygen': [],
            'damage': [],
            'waste': [],
            'growth': []
        }
        
        self.obs = self.env.reset()
        self.done = False
        self.step_count = 0
    
    def run_simulation(self):
        """Simülasyonu çalıştır ve sonuçları göster"""
        print("=" * 60)
        print("🧬 HÜCRE SİMÜLASYONU BAŞLIYOR")
        print("=" * 60)
        
        while not self.done and self.step_count < self.env.max_steps:
            # Aksiyon seç
            if self.model:
                action, _ = self.model.predict(self.obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            
            # Adım at
            self.obs, reward, self.done, info = self.env.step(action)
            self.step_count += 1
            
            # Veriyi kaydet
            self.record_data()
            
            # Her 50 adımda rapor
            if self.step_count % 50 == 0:
                self.print_status()
        
        # Final rapor
        print("\n" + "=" * 60)
        print("🏁 SİMÜLASYON SONUÇLARI")
        print("=" * 60)
        self.print_final_report()
        
        # Grafikleri göster
        self.plot_results()
    
    def record_data(self):
        """Veriyi kaydet"""
        cell = self.env.cell
        self.history['step'].append(self.step_count)
        self.history['size'].append(cell.size)
        self.history['atp'].append(cell.atp)
        self.history['glucose'].append(cell.glucose)
        self.history['oxygen'].append(cell.oxygen)
        self.history['damage'].append(cell.damage)
        waste = cell.co2 + cell.lactate + cell.urea
        self.history['waste'].append(waste)
        self.history['growth'].append(cell.growth_points)
    
    def print_status(self):
        """Anlık durum raporu"""
        cell = self.env.cell
        status = "✓ CANLI" if cell.is_alive() else "✗ ÖLÜ"
        print(f"Adım {self.step_count}: {status} | "
              f"Boyut: {cell.size:.2f} | "
              f"ATP: {cell.atp:.1f} | "
              f"Hasar: {cell.damage:.1f}% | "
              f"Büyüme: {cell.growth_points:.1f}%")
    
    def print_final_report(self):
        """Final raporu"""
        cell = self.env.cell
        max_size = max(self.history['size'])
        avg_atp = np.mean(self.history['atp'][-100:]) if len(self.history['atp']) > 100 else np.mean(self.history['atp'])
        
        print(f"Toplam Adım: {self.step_count}")
        print(f"Final Durum: {'✓ CANLI' if cell.is_alive() else '✗ ÖLÜ'}")
        print(f"Final Boyut: {cell.size:.2f}x")
        print(f"Maksimum Boyut: {max_size:.2f}x")
        print(f"Final Yaş: {cell.age} adım")
        print(f"Ortalama ATP (son 100): {avg_atp:.1f}")
        print(f"Final Hasar: {cell.damage:.1f}%")
        print(f"Final Stres: {cell.stress_level:.1f}%")
        
        if cell.can_divide():
            print("\n🎉 BAŞARI: Hücre bölünmeye hazır!")
        elif max_size >= 1.5:
            print("\n👍 İYİ: Hücre iyi büyüdü!")
        elif self.step_count >= 200:
            print("\n😊 ORTA: Hücre uzun süre hayatta kaldı")
        else:
            print("\n😞 BAŞARISIZ: Hücre erken öldü")
    
    def plot_results(self):
        """Sonuçları görselleştir"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🧬 Hücre Simülasyonu Sonuçları', fontsize=16, fontweight='bold')
        
        steps = self.history['step']
        
        # 1. Boyut
        axes[0, 0].plot(steps, self.history['size'], 'g-', linewidth=2)
        axes[0, 0].set_title('Hücre Boyutu', fontweight='bold')
        axes[0, 0].set_xlabel('Adım')
        axes[0, 0].set_ylabel('Boyut (x)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.8, color='r', linestyle='--', alpha=0.5, label='Bölünme Eşiği')
        axes[0, 0].legend()
        
        # 2. ATP
        axes[0, 1].plot(steps, self.history['atp'], 'gold', linewidth=2)
        axes[0, 1].set_title('ATP (Enerji)', fontweight='bold')
        axes[0, 1].set_xlabel('Adım')
        axes[0, 1].set_ylabel('ATP')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Kritik Seviye')
        axes[0, 1].legend()
        
        # 3. Glikoz
        axes[0, 2].plot(steps, self.history['glucose'], 'brown', linewidth=2)
        axes[0, 2].set_title('Glikoz', fontweight='bold')
        axes[0, 2].set_xlabel('Adım')
        axes[0, 2].set_ylabel('Glikoz')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Oksijen
        axes[1, 0].plot(steps, self.history['oxygen'], 'cyan', linewidth=2)
        axes[1, 0].set_title('Oksijen', fontweight='bold')
        axes[1, 0].set_xlabel('Adım')
        axes[1, 0].set_ylabel('Oksijen')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Kritik Seviye')
        axes[1, 0].legend()
        
        # 5. Hasar
        axes[1, 1].plot(steps, self.history['damage'], 'red', linewidth=2)
        axes[1, 1].set_title('Hasar', fontweight='bold')
        axes[1, 1].set_xlabel('Adım')
        axes[1, 1].set_ylabel('Hasar (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=100, color='darkred', linestyle='--', alpha=0.5, label='Ölümcül')
        axes[1, 1].legend()
        
        # 6. Atık
        axes[1, 2].plot(steps, self.history['waste'], 'purple', linewidth=2)
        axes[1, 2].set_title('Toplam Atık', fontweight='bold')
        axes[1, 2].set_xlabel('Adım')
        axes[1, 2].set_ylabel('Atık')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Tehlikeli Seviye')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Kaydet
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/simulation_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 Grafikler kaydedildi: plots/simulation_results.png")
        
        plt.show()

def compare_models():
    """Eğitilmiş model vs Rastgele karşılaştırma"""
    print("=" * 60)
    print("🔬 MODEL KARŞILAŞTIRMASI")
    print("=" * 60)
    
    results = {}
    
    # 1. Eğitilmiş model
    print("\n1️⃣ Eğitilmiş Model Test Ediliyor...")
    if os.path.exists("models/cell_model.zip"):
        vis = CellVisualizer(model_path="models/cell_model", max_steps=500)
        vis.run_simulation()
        results['trained'] = {
            'steps': vis.step_count,
            'max_size': max(vis.history['size']),
            'alive': vis.env.cell.is_alive()
        }
    else:
        print("❌ Model bulunamadı!")
        results['trained'] = None
    
    # 2. Rastgele aksiyonlar
    print("\n2️⃣ Rastgele Aksiyonlar Test Ediliyor...")
    vis = CellVisualizer(model_path=None, max_steps=500)
    vis.run_simulation()
    results['random'] = {
        'steps': vis.step_count,
        'max_size': max(vis.history['size']),
        'alive': vis.env.cell.is_alive()
    }
    
    # Karşılaştırma
    print("\n" + "=" * 60)
    print("📊 KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 60)
    
    if results['trained']:
        print(f"\n🤖 Eğitilmiş Model:")
        print(f"  Yaşadığı Süre: {results['trained']['steps']} adım")
        print(f"  Max Boyut: {results['trained']['max_size']:.2f}x")
        print(f"  Son Durum: {'✓ Canlı' if results['trained']['alive'] else '✗ Öldü'}")
    
    print(f"\n🎲 Rastgele Aksiyonlar:")
    print(f"  Yaşadığı Süre: {results['random']['steps']} adım")
    print(f"  Max Boyut: {results['random']['max_size']:.2f}x")
    print(f"  Son Durum: {'✓ Canlı' if results['random']['alive'] else '✗ Öldü'}")
    
    if results['trained']:
        improvement = (results['trained']['steps'] / results['random']['steps'] - 1) * 100
        print(f"\n📈 İyileştirme: {improvement:+.1f}%")

def main():
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "compare":
            # Model vs rastgele karşılaştırma
            compare_models()
        elif command == "random":
            # Sadece rastgele
            vis = CellVisualizer(model_path=None, max_steps=500)
            vis.run_simulation()
        else:
            print("Kullanım:")
            print("  python visualize.py          - Model ile simülasyon")
            print("  python visualize.py compare  - Model vs Rastgele")
            print("  python visualize.py random   - Sadece rastgele")
    else:
        # Varsayılan: model ile simülasyon
        vis = CellVisualizer(model_path="models/cell_model", max_steps=500)
        vis.run_simulation()

if __name__ == "__main__":
    main()