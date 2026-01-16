# test_cell.py - Hücre Sistemini Test Et

from cell import Cell
from environment import CellEnv
import numpy as np

def test_cell_basic():
    """Temel hücre fonksiyonlarını test et"""
    print("=" * 60)
    print("🧪 TEMEL HÜCRE TESTLERİ")
    print("=" * 60)
    
    cell = Cell()
    print("\n1. Başlangıç Durumu:")
    print(cell)
    print(f"Canlı mı? {cell.is_alive()}")
    print(f"Metabolik Hız: {cell.get_metabolic_rate():.2f}x")
    print(f"Bazal Maliyet: {cell.get_maintenance_cost():.2f} ATP/adım")
    
    print("\n2. Aerobik Solunum Testi:")
    initial_atp = cell.atp
    success = cell.aerobic_respiration()
    print(f"Başarılı: {success}")
    print(f"ATP Değişimi: {initial_atp:.1f} -> {cell.atp:.1f} ({cell.atp - initial_atp:+.1f})")
    
    print("\n3. Protein Sentezi Testi:")
    initial_protein = cell.structural_proteins
    success = cell.protein_synthesis()
    print(f"Başarılı: {success}")
    print(f"Protein: {initial_protein:.1f} -> {cell.structural_proteins:.1f}")
    
    print("\n4. Bazal Metabolizma Testi (5 adım):")
    for i in range(5):
        cell.passive_processes()
        print(f"Adım {i+1}: ATP={cell.atp:.1f}, Yaş={cell.age}, Hasar={cell.damage:.1f}%")
    
    print("\n5. Büyüme Testi:")
    cell.growth_points = 95
    print(f"Büyüme Puanı: {cell.growth_points:.1f}%")
    print(f"Mevcut Boyut: {cell.size:.2f}x")
    cell.passive_processes()
    print(f"Yeni Büyüme Puanı: {cell.growth_points:.1f}%")
    print(f"Yeni Boyut: {cell.size:.2f}x")
    
    print("\n✅ Temel testler tamamlandı!")

def test_cell_death():
    """Ölüm koşullarını test et"""
    print("\n" + "=" * 60)
    print("💀 ÖLÜM KOŞULLARI TESTLERİ")
    print("=" * 60)
    
    # Test 1: ATP bitince ölüm
    print("\n1. ATP Bitince:")
    cell = Cell()
    cell.atp = 1
    print(f"ATP: {cell.atp:.1f}, Canlı: {cell.is_alive()}")
    cell.atp = 0
    print(f"ATP: {cell.atp:.1f}, Canlı: {cell.is_alive()}")
    
    # Test 2: Oksijen bitince ölüm
    print("\n2. Oksijen Bitince:")
    cell = Cell()
    cell.oxygen = 1
    print(f"Oksijen: {cell.oxygen:.1f}, Canlı: {cell.is_alive()}")
    cell.oxygen = 0
    print(f"Oksijen: {cell.oxygen:.1f}, Canlı: {cell.is_alive()}")
    
    # Test 3: Hasar 100 olunca ölüm
    print("\n3. Hasar 100 Olunca:")
    cell = Cell()
    cell.damage = 99
    print(f"Hasar: {cell.damage:.1f}%, Canlı: {cell.is_alive()}")
    cell.damage = 100
    print(f"Hasar: {cell.damage:.1f}%, Canlı: {cell.is_alive()}")
    
    print("\n✅ Ölüm testleri tamamlandı!")

def test_environment():
    """Environment'ı test et"""
    print("\n" + "=" * 60)
    print("🌍 ENVIRONMENT TESTLERİ")
    print("=" * 60)
    
    env = CellEnv(max_steps=100)
    
    print("\n1. Ortam Oluşturuldu:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    print("\n2. Reset Testi:")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    print("\n3. Tek Adım Testi:")
    action = 0  # Aerobik solunum
    obs, reward, done, info = env.step(action)
    print(f"Aksiyon: {env.get_action_name(action)}")
    print(f"Ödül: {reward:.2f}")
    print(f"Bitti: {done}")
    print(f"Info: {info}")
    
    print("\n4. 10 Adım Simülasyonu:")
    env.reset()
    total_reward = 0
    for step in range(10):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Adım {step+1}: {env.get_action_name(action)}, "
              f"Ödül: {reward:.2f}, ATP: {env.cell.atp:.1f}")
        if done:
            print("Episode bitti!")
            break
    
    print(f"\nToplam Ödül: {total_reward:.2f}")
    print("\n✅ Environment testleri tamamlandı!")

def test_metabolic_pathways():
    """Metabolik yolları test et"""
    print("\n" + "=" * 60)
    print("⚗️ METABOLİK YOLLAR TESTLERİ")
    print("=" * 60)
    
    cell = Cell()
    
    pathways = [
        ("Aerobik Solunum", lambda: cell.aerobic_respiration()),
        ("Anaerobik Glikoliz", lambda: cell.anaerobic_glycolysis()),
        ("Yağ Oksidasyonu", lambda: cell.fat_oxidation()),
        ("Protein Sentezi", lambda: cell.protein_synthesis()),
        ("Detoksifikasyon", lambda: cell.detoxification()),
        ("Onarım", lambda: cell.repair_damage()),
    ]
    
    for name, func in pathways:
        cell.reset()
        cell.damage = 20  # Onarım testi için
        print(f"\n{name}:")
        print(f"  Öncesi: ATP={cell.atp:.1f}, Glikoz={cell.glucose:.1f}")
        success = func()
        print(f"  Başarı: {success}")
        print(f"  Sonrası: ATP={cell.atp:.1f}, Glikoz={cell.glucose:.1f}")
    
    print("\n✅ Metabolik yol testleri tamamlandı!")

def test_growth_mechanics():
    """Büyüme mekaniğini test et"""
    print("\n" + "=" * 60)
    print("📈 BÜYÜME MEKANİĞİ TESTİ")
    print("=" * 60)
    
    cell = Cell()
    print(f"\nBaşlangıç Boyutu: {cell.size:.2f}x")
    print(f"Başlangıç Metabolik Hız: {cell.get_metabolic_rate():.3f}x")
    
    # Hızlı büyüme simülasyonu
    print("\nBüyüme simülasyonu (100 adım):")
    for i in range(100):
        # Güçlü enerji üretimi
        cell.aerobic_respiration()
        cell.protein_synthesis()
        cell.membrane_synthesis()
        
        cell.passive_processes()
        
        if i % 20 == 0:
            print(f"Adım {i}: Boyut={cell.size:.2f}x, "
                  f"Büyüme={cell.growth_points:.1f}%, "
                  f"Metabolik={cell.get_metabolic_rate():.3f}x")
    
    print(f"\nFinal Boyut: {cell.size:.2f}x")
    print(f"Final Metabolik Hız: {cell.get_metabolic_rate():.3f}x")
    print(f"Bölünebilir mi? {cell.can_divide()}")
    
    print("\n✅ Büyüme testleri tamamlandı!")

def run_all_tests():
    """Tüm testleri çalıştır"""
    print("\n" + "=" * 60)
    print("🔬 TÜM TESTLER BAŞLIYOR")
    print("=" * 60)
    
    test_cell_basic()
    test_cell_death()
    test_environment()
    test_metabolic_pathways()
    test_growth_mechanics()
    
    print("\n" + "=" * 60)
    print("✅ TÜM TESTLER TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        tests = {
            'basic': test_cell_basic,
            'death': test_cell_death,
            'env': test_environment,
            'metabolic': test_metabolic_pathways,
            'growth': test_growth_mechanics,
            'all': run_all_tests
        }
        
        if test_name in tests:
            tests[test_name]()
        else:
            print("Mevcut testler:")
            print("  python test_cell.py basic      - Temel testler")
            print("  python test_cell.py death      - Ölüm testleri")
            print("  python test_cell.py env        - Environment testleri")
            print("  python test_cell.py metabolic  - Metabolik yol testleri")
            print("  python test_cell.py growth     - Büyüme testleri")
            print("  python test_cell.py all        - Tüm testler")
    else:
        run_all_tests()