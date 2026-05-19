import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point, MultiPolygon
from matplotlib.widgets import Button, TextBox
from scipy.spatial import Voronoi
import copy

# --- SİMÜLASYON SABİTLERİ ---
ITERASYON_SAYISI = 100
ANIMASYON_HIZI_MS = 50
MIN_KOSSE_SAYISI = 3 
MIN_DRONE_SAYISI = 1 
MAX_DRONE_SAYISI = 200 
INITIAL_DRONE_COUNT = 30

class Environment:
    def __init__(self, boundary_coords, obstacles_coords_list=None):
        self.boundary_coords = np.array(boundary_coords)
        self.polygon = Polygon(self.boundary_coords)
        self.navigable_area = self.polygon
        self.obstacles = []
        
        if obstacles_coords_list:
            for obs_coords in obstacles_coords_list:
                obs_poly = Polygon(obs_coords)
                self.obstacles.append(obs_poly)
                self.navigable_area = self.navigable_area.difference(obs_poly)
        
        rep_point = self.navigable_area.representative_point()
        self.centroid = np.array([rep_point.x, rep_point.y])
        
        min_x, min_y, max_x, max_y = self.polygon.bounds
        uzaklik = max(max_x - min_x, max_y - min_y) * 10
        self.ghost_points = np.array([
            [min_x - uzaklik, min_y - uzaklik],
            [max_x + uzaklik, min_y - uzaklik],
            [max_x + uzaklik, max_y + uzaklik],
            [min_x - uzaklik, max_y + uzaklik]
        ])

class BaseSolver:
    def __init__(self, env, drone_count, initial_positions, colors):
        self.env = env
        self.drone_count = drone_count
        self.positions = copy.deepcopy(initial_positions)
        self.colors = colors

    def get_cell_areas(self):
        """Her dronun kapsadığı güncel alan miktarını hesaplar."""
        areas = []
        pts = np.vstack((self.positions, self.env.ghost_points))
        vor = Voronoi(pts)
        for i in range(self.drone_count):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or not region: 
                areas.append(0)
                continue
            poly = Polygon([vor.vertices[v] for v in region])
            try:
                # Engelsiz alanla kesişim
                inter = self.env.navigable_area.intersection(poly)
                areas.append(inter.area if not inter.is_empty else 0)
            except:
                areas.append(0)
        return np.array(areas)

    def draw(self, ax, title):
        ax.clear()
        min_x, min_y, max_x, max_y = self.env.polygon.bounds
        
        # 1. Şeklin tam merkezini buluyoruz
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        # 2. Şeklin en uzun kenarını bulup, her iki ekseni de bu uzunluğa eşitliyoruz
        max_dim = max(max_x - min_x, max_y - min_y)
        half_range = (max_dim / 2.0) * 1.05  # %5 rahatlama boşluğu eklendi
        
        # 3. Hem X hem de Y eksenini BİREBİR AYNI boyuta zorluyoruz (Kare Kutu)
        ax.set_xlim(center_x - half_range, center_x + half_range)
        ax.set_ylim(center_y - half_range, center_y + half_range)
        
        ax.plot(*self.env.polygon.exterior.xy, color='black', linewidth=2)
        for obs in self.env.obstacles:
            ax.fill(*obs.exterior.xy, color='black', alpha=0.6, hatch='//')
        
        # Voronoi Hücrelerini Çiz
        pts = np.vstack((self.positions, self.env.ghost_points))
        vor = Voronoi(pts)
        for i in range(self.drone_count):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or not region: continue
            try:
                poly = Polygon([vor.vertices[v] for v in region])
                clipped = self.env.navigable_area.intersection(poly)
                if not clipped.is_empty:
                    if isinstance(clipped, MultiPolygon):
                        for p in clipped.geoms:
                            ax.fill(*p.exterior.xy, color=self.colors[i], alpha=0.5, edgecolor='white', lw=0.5)
                    else:
                        ax.fill(*clipped.exterior.xy, color=self.colors[i], alpha=0.5, edgecolor='white', lw=0.5)
            except: pass

        ax.scatter(self.positions[:, 0], self.positions[:, 1], c=self.colors, s=30, edgecolors='black', zorder=10)
        ax.set_title(title, fontsize=10)
        
        # 4. EN ÖNEMLİ KISIM: Geometrik esnemeyi yasakla ve 1:1 oranında kilitle
        ax.set_aspect('equal', adjustable='box')

class APFSolver(BaseSolver):
    def __init__(self, env, drone_count, initial_positions, colors):
        super().__init__(env, drone_count, initial_positions, colors)
        self.velocities = np.zeros((self.drone_count, 2))
        
        # SENİN ORİJİNAL PARAMETRELERİN
        self.DT = 0.04
        self.DAMPING = 0.6          
        self.K_NOISE = 0.01
        self.K_ATTRACT = 0.25       
        self.K_REPEL_AGENT = 0.12   
        self.REPULSION_DIST = 0.35 
        self.K_REPEL_BOUND = 0.4 
        self.BOUND_MARGIN = 0.15

    def step(self):
        forces = np.zeros((self.drone_count, 2))
        
        for i in range(self.drone_count):
            pos_i = self.positions[i]
            p_point = Point(pos_i)
            
            # 1. Çekim Kuvveti
            vec_to_center = self.env.centroid - pos_i
            forces[i] += self.K_ATTRACT * vec_to_center
            
            # 2. Ajanlar Arası İtme
            for j in range(self.drone_count):
                if i == j: continue
                vec_ij = pos_i - self.positions[j]
                dist_ij = np.linalg.norm(vec_ij)
                if dist_ij < self.REPULSION_DIST:
                    magnitude = self.K_REPEL_AGENT * (1.0 / (max(dist_ij, 0.04)))
                    forces[i] += (vec_ij / (dist_ij + 1e-6)) * magnitude

            # 3. Dış Sınır İtmesi
            dist_to_boundary = p_point.distance(self.env.polygon.exterior)
            is_inside = self.env.polygon.contains(p_point)
            
            if is_inside and dist_to_boundary < self.BOUND_MARGIN:
                closest_p = self.env.polygon.exterior.interpolate(self.env.polygon.exterior.project(p_point))
                vec_from_boundary = pos_i - np.array(closest_p.coords[0])
                dist = np.linalg.norm(vec_from_boundary)
                magnitude = self.K_REPEL_BOUND * (1.0 / (dist_to_boundary + 0.01))
                forces[i] += (vec_from_boundary / (dist + 1e-6)) * magnitude
            elif not is_inside:
                forces[i] += (self.env.centroid - pos_i) * 5.0

            # 4. İÇ ENGELLERDEN (Obstacles) İTME
            for obs in self.env.obstacles:
                dist_to_obs = p_point.distance(obs)
                if dist_to_obs < self.BOUND_MARGIN:
                    closest_p = obs.exterior.interpolate(obs.exterior.project(p_point))
                    vec_from_obs = pos_i - np.array(closest_p.coords[0])
                    dist = np.linalg.norm(vec_from_obs)
                    magnitude = self.K_REPEL_BOUND * (1.0 / (dist_to_obs + 0.01))
                    forces[i] += (vec_from_obs / (dist + 1e-6)) * magnitude

        # Senin orijinal gürültü ve hız hesabın
        forces += (np.random.rand(self.drone_count, 2) - 0.5) * self.K_NOISE
        self.velocities = (self.velocities + forces * self.DT) * self.DAMPING
        self.positions += self.velocities * self.DT
        
        # KATI SINIR VE ENGEL KONTROLÜ
        for i in range(self.drone_count):
            p_point = Point(self.positions[i])
            if not self.env.polygon.contains(p_point):
                closest = self.env.polygon.exterior.interpolate(self.env.polygon.exterior.project(p_point))
                self.positions[i] = np.array(closest.coords[0])
                self.velocities[i] = np.zeros(2)
            else:
                for obs in self.env.obstacles:
                    if obs.contains(p_point):
                        closest = obs.exterior.interpolate(obs.exterior.project(p_point))
                        self.positions[i] = np.array(closest.coords[0])
                        self.velocities[i] = np.zeros(2)
                        break

class LloydSolver(BaseSolver):
    def step(self):
        pts = np.vstack((self.positions, self.env.ghost_points))
        vor = Voronoi(pts)
        for i in range(self.drone_count):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or not region: continue
            try:
                poly = Polygon([vor.vertices[v] for v in region])
                clipped = self.env.navigable_area.intersection(poly)
                if not clipped.is_empty and clipped.area > 0:
                    new_pos = clipped.centroid
                    # Eğer şeklin ağırlık merkezi dışarıya veya engelin içine düştüyse:
                    if not self.env.navigable_area.contains(new_pos):
                        # Shapely'nin alan içinde kalmayı garanti eden temsilci noktasını kullan
                        new_pos = clipped.representative_point()
                    
                    self.positions[i] = np.array(new_pos.coords[0])
            except: pass

class SimulationManager:
    def __init__(self):
        self.drone_count = INITIAL_DRONE_COUNT
        self.boundary_coords = None
        self.obstacles_coords_list = []
        self.lloyd_history = []
        self.apf_history = []
        
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title('Drone Dağılım Analizi')
        
    def setup_ui(self):
        self.ax_setup = self.fig.add_subplot(111)
        self.ax_setup.set_title("1. Drone Sayısı Girin -> 2. Alanı Çiz Butonuna Basın -> 3. Köşeleri Tıklayın (ENTER)", fontsize=12)

        self.fig.subplots_adjust(bottom=0.15)
        # 1. ax_box yerine self.ax_box yapıyoruz
        self.ax_box = self.fig.add_axes([0.3, 0.04, 0.1, 0.05])
        self.text_box = TextBox(self.ax_box, 'Drone: ', initial=str(self.drone_count))
        
        # 2. ax_btn yerine self.ax_btn yapıyoruz
        self.ax_btn = self.fig.add_axes([0.45, 0.04, 0.2, 0.05])
        self.btn = Button(self.ax_btn, 'Alanı Tanımla', color='lightblue')
        
        def start_setup(event):
            self.drone_count = int(self.text_box.text)
            
            # 3. İçeriden artık self üzerinden sorunsuzca çağırıyoruz
            self.ax_btn.set_visible(False)
            self.ax_box.set_visible(False)
            
            self.ax_setup.set_title("ADIM 1: Dış Sınırı Çizin (Bitirmek için ENTER'a basın)", 
                                    fontsize=14, color='red', fontweight='bold')
            self.fig.canvas.draw()
            
            coords = plt.ginput(n=-1, timeout=0, show_clicks=True)
            if len(coords) >= 3:
                self.boundary_coords = coords
                self.ax_setup.plot(*Polygon(coords).exterior.xy, 'k-')
                
                while True:
                    self.ax_setup.set_title("ADIM 2: Engel Çizin (Pas geçmek veya Bitirmek için doğrudan ENTER)", 
                                            fontsize=14, color='darkorange', fontweight='bold')
                    self.fig.canvas.draw()
                    
                    obs = plt.ginput(n=-1, timeout=0, show_clicks=True)
                    if len(obs) < 3: break
                    self.obstacles_coords_list.append(obs)
                    self.ax_setup.fill(*Polygon(obs).exterior.xy, color='red', alpha=0.3)
                
                self.run_sim()

        self.btn.on_clicked(start_setup)
        plt.show()

    def run_sim(self):
        self.fig.clf()
        
        # Ekranı bölmek için yeni GridSpec ayarları
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.3)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])      # Üst Sol
        self.ax2 = self.fig.add_subplot(gs[0, 1])      # Üst Sağ
        self.ax_graph = self.fig.add_subplot(gs[1, :]) # Alt satırın tamamı
        
        self.fig.subplots_adjust(bottom=0.08, left=0.05, right=0.95, top=0.92)

        env = Environment(self.boundary_coords, self.obstacles_coords_list)
        colors = plt.cm.viridis(np.linspace(0, 1, self.drone_count))
        
        # Dronları sadece engelsiz ve geçerli alanın (navigable_area) içine yerleştir
        init_pos = []
        min_x, min_y, max_x, max_y = env.navigable_area.bounds
        while len(init_pos) < self.drone_count:
            # Alanın genel sınırları içinde rastgele bir nokta seç
            rx = np.random.uniform(min_x, max_x)
            ry = np.random.uniform(min_y, max_y)
            p = Point(rx, ry)
            # Nokta gerçekten haritanın içindeyse ve engellere çarpmıyorsa listeye al
            if env.navigable_area.contains(p):
                init_pos.append([rx, ry])
        init_pos = np.array(init_pos)
        
        self.lloyd = LloydSolver(env, self.drone_count, init_pos, colors)
        self.apf = APFSolver(env, self.drone_count, init_pos, colors)

        def update(frame):
            self.lloyd.step()
            self.apf.step()
            
            self.lloyd.draw(self.ax1, f"Lloyd (Voronoi) - İterasyon: {frame}")
            self.apf.draw(self.ax2, f"Potansiyel Alan (APF) - İterasyon: {frame}")
            
            # Verimlilik Analizi (Standart Sapma)
            l_areas = self.lloyd.get_cell_areas()
            a_areas = self.apf.get_cell_areas()
            
            self.lloyd_history.append(np.std(l_areas))
            self.apf_history.append(np.std(a_areas))
            
            self.ax_graph.clear()
            self.ax_graph.plot(self.lloyd_history, label='Lloyd', color='blue', lw=2)
            self.ax_graph.plot(self.apf_history, label='APF', color='red', lw=2)
            self.ax_graph.set_title("Dağılım Verimliliği (Alan Standart Sapması)")
            self.ax_graph.set_ylabel("Standart Sapma ($\\sigma$)")
            self.ax_graph.set_xlabel("Adım")
            self.ax_graph.legend()
            self.ax_graph.grid(True, alpha=0.3)

        self.ani = FuncAnimation(self.fig, update, frames=ITERASYON_SAYISI, interval=ANIMASYON_HIZI_MS, repeat=False)
        self.fig.canvas.draw()

if __name__ == "__main__":
    SimulationManager().setup_ui()
