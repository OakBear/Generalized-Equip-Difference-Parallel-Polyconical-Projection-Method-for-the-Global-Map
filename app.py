"""广义等差分纬线多圆锥投影 交互式可视化
基于郝晓光论文 (Generalized Equip-Difference Parallel Polyconical Projection)
"""
import io, time, textwrap
import streamlit as st
import plotly.graph_objects as go
import shapefile
import numpy as np
from numpy import sin, cos, arcsin, arctan2, radians, degrees

# ========== 常量 ==========
OCEAN       = '#dceaf5'
GRID_C      = '#b8d0e0'
EQ_C        = '#7fb0d0'
BORDER      = '#666680'
LAND_COLORS = ['#dfe8c0', '#daceab', '#ccc3ac', '#dbd5af',
               '#cbd8b8', '#e3d0b0', '#d6c9ab']
DATA_DIR   = r"c:/desktopp/造船/ne_temp"
COUNTRIES  = f"{DATA_DIR}/ne_110m_admin_0_countries.shp"
CITIES     = f"{DATA_DIR}/ne_110m_populated_places.shp"

# ========== 数据加载 ==========
@st.cache_data(show_spinner="加载地理数据...")
def load_countries():
    sf = shapefile.Reader(COUNTRIES)
    shapes, recs = sf.shapes(), sf.records()
    out = []
    for shp, rec in zip(shapes, recs):
        try:
            cidx = int(rec[31]) % len(LAND_COLORS)
        except (ValueError, TypeError):
            cidx = 0
        parts = list(shp.parts) + [len(shp.points)]
        rings = []
        for i in range(len(parts) - 1):
            pts = shp.points[parts[i]:parts[i+1]]
            if len(pts) >= 3:
                rings.append(np.array(pts, dtype=float))
        out.append(dict(name=rec[18], rings=rings, color=cidx))
    sf.close()
    return out

@st.cache_data(show_spinner="加载城市数据...")
def load_cities():
    sf = shapefile.Reader(CITIES)
    ff = sf.fields[1:]
    def idx(name):
        for i, f in enumerate(ff):
            if f[0] == name:
                return i
        return None
    ni, li, lti = idx('NAME'), idx('LONGITUDE'), idx('LATITUDE')
    out = []
    for r in sf.records():
        try:
            out.append(dict(name=r[ni], lon=float(r[li]), lat=float(r[lti])))
        except (ValueError, TypeError):
            pass
    sf.close()
    return out

# ========== 球面旋转 → 等距圆柱投影 ==========
def transform(lons, lats, pole_lon, central_mer):
    """
    球面旋转 → 等距圆柱投影。
    旋转使赤道上 (pole_lon, 0°) 成为新北极。
    central_mer 仅用于最终投影的水平居中，不参与旋转。
    """
    pl = radians(pole_lon)
    l, p = radians(lons), radians(lats)
    # 原始笛卡尔坐标 (不中心化)
    x = cos(p) * cos(l)
    y = cos(p) * sin(l)
    z = sin(p)
    # 旋转: Ry(-90°) * Rz(-pl) 将 (pole_lon, 0°) 转到北极
    # x' = -z
    # y' = -sin(pl)*x + cos(pl)*y  =  cos(p)*sin(l-pl)
    # z' =  cos(pl)*x + sin(pl)*y  =  cos(p)*cos(l-pl)
    xr = -z
    yr = -sin(pl) * x + cos(pl) * y
    zr = cos(pl) * x + sin(pl) * y
    lat_r = degrees(arcsin(np.clip(zr, -1.0, 1.0)))
    # 避免 -0.0 导致 atan2 异常 (极点处广义经度任意，设为 0)
    xr_safe = np.where(np.abs(xr) < 1e-15, 0.0, xr)
    lon_r = degrees(arctan2(yr, xr_safe))
    # 水平居中 + 回绕到 [-180, 180]
    lon_r = lon_r - central_mer
    lon_r = ((lon_r + 180) % 360) - 180
    return lon_r, lat_r

def split_at_edge(xs, ys, threshold=160):
    """若 x 跳变 > threshold 则插入 None 断开线段"""
    ox, oy = [], []
    for i in range(len(xs)):
        if i > 0 and abs(xs[i] - xs[i-1]) > threshold:
            ox.append(None); oy.append(None)
        ox.append(xs[i]); oy.append(ys[i])
    return ox, oy

# ========== Streamlit UI ==========
st.set_page_config(layout="wide", page_title="广义等差分纬线多圆锥投影")

# ---------- side bar ----------
with st.sidebar:
    st.markdown("## 投影参数")
    pole_lon = st.slider("极点经度 (新北极在赤道的位置)", -180, 180, 90, 1,
                         help="论文推荐 60°（落入印度洋）; 90° 使常规北极居中")
    central_m = st.slider("中央经线 (地图左右居中位置)", -180, 180, 180, 1,
                          help="180° 使北极落在 x=0")
    show_grid = st.checkbox("经纬网", True)
    grid_int = st.select_slider("网线间隔 (°)", [15, 30, 45, 90], 30)
    grid_w = st.slider("经纬线线宽", 0.2, 4.0, 1.0, 0.1)
    show_cities = st.checkbox("城市点", False)
    show_fill = st.checkbox("陆地填充", True)
    border_w = st.slider("边界线宽", 0.0, 2.0, 0.6, 0.1)
    st.markdown("---")
    st.caption("**论文推荐**: 极点 60° (落入海洋)  ·  默认 90°/180° 为环绕北冰洋版")

# ---------- load data ----------
countries = load_countries()
cities = load_cities() if show_cities else []

# ---------- build figure ----------
fig = go.Figure()

# 海洋由 plot_bgcolor 提供，不额外绘制海洋多边形
# ---------- 国家边界 ----------
n_rings = 0
for c in countries:
    for ring in c['rings']:
        n_rings += 1
        prx, pry = transform(ring[:, 0], ring[:, 1], pole_lon, central_m)
        xs, ys = split_at_edge(prx, pry)

        if show_fill:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                fill='toself',
                fillcolor=LAND_COLORS[c['color']],
                line=dict(color=BORDER, width=border_w),
                showlegend=False,
                hovertext=c['name'],
                hoverinfo='text',
                name=c['name']
            ))
        else:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=BORDER, width=border_w),
                showlegend=False,
                hovertext=c['name'],
                hoverinfo='text',
                name=c['name']
            ))

# ---------- 经纬网 ----------
if show_grid:
    grid_lons = range(-180, 181, grid_int)
    grid_lats = range(-90, 91, grid_int)

    # 经线 (从南到北的竖线)
    for lon in grid_lons:
        ring_lon = np.full(91, lon, float)
        ring_lat = np.arange(-90, 91, 2, float)
        gx, gy = transform(ring_lon, ring_lat, pole_lon, central_m)
        gxs, gys = split_at_edge(gx, gy)
        fig.add_trace(go.Scatter(
            x=gxs, y=gys, mode='lines',
            line=dict(color=GRID_C, width=grid_w, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

    # 纬线 (从左到右的横线)
    for lat in grid_lats:
        ring_lon = np.arange(-180, 181, 2, float)
        ring_lat = np.full_like(ring_lon, lat, float)
        gx, gy = transform(ring_lon, ring_lat, pole_lon, central_m)
        gxs, gys = split_at_edge(gx, gy)
        fig.add_trace(go.Scatter(
            x=gxs, y=gys, mode='lines',
            line=dict(color=GRID_C, width=grid_w, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

    # 赤道加粗
    eq_lon = np.arange(-180, 181, 2, float)
    eq_lat = np.zeros_like(eq_lon)
    gx, gy = transform(eq_lon, eq_lat, pole_lon, central_m)
    gxs, gys = split_at_edge(gx, gy)
    fig.add_trace(go.Scatter(
        x=gxs, y=gys, mode='lines',
        line=dict(color=EQ_C, width=grid_w * 2),
        showlegend=False, hoverinfo='skip'
    ))

# ---------- 城市 ----------
if show_cities and cities:
    lons = np.array([c['lon'] for c in cities])
    lats = np.array([c['lat'] for c in cities])
    cx, cy = transform(lons, lats, pole_lon, central_m)
    hover_texts = [c['name'] for c in cities]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode='markers',
        marker=dict(color='#dc3c3c', size=4,
                    line=dict(width=0.4, color='white')),
        text=hover_texts, hoverinfo='text',
        showlegend=False
    ))

# ---------- layout ----------
fig.update_layout(
    height=750,
    template='none',
    paper_bgcolor='white',
    plot_bgcolor=OCEAN,
    xaxis=dict(visible=False, scaleanchor='y', constrain='domain',
               range=[-185, 185], showgrid=False),
    yaxis=dict(visible=False, range=[-95, 95], showgrid=False),
    margin=dict(l=5, r=5, t=5, b=5),
    hovermode='closest',
)

# 圆角遮罩 (四个角用 paper_bgcolor 遮盖)
cr = 8
xm, xM, ym, yM = -185, 185, -95, 95
for path in [
    f'M {xm},{yM-cr} L {xm},{yM} L {xm+cr},{yM} A {cr},{cr} 0 0,1 {xm},{yM-cr} Z',
    f'M {xM-cr},{yM} L {xM},{yM} L {xM},{yM-cr} A {cr},{cr} 0 0,1 {xM-cr},{yM} Z',
    f'M {xM},{ym+cr} L {xM},{ym} L {xM-cr},{ym} A {cr},{cr} 0 0,1 {xM},{ym+cr} Z',
    f'M {xm+cr},{ym} L {xm},{ym} L {xm},{ym+cr} A {cr},{cr} 0 0,1 {xm+cr},{ym} Z',
]:
    fig.add_shape(type='path', path=path, layer='above',
                  fillcolor='white', line=dict(width=0))

# ---------- 显示 ----------
st.plotly_chart(fig, width='stretch')

# 圆角 CSS (补充容器级圆角)
st.markdown("""
<style>
.stPlotlyChart {
    border-radius: 20px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------- 状态面板 ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("国家/地区", len(countries))
col2.metric("多边形环", n_rings)
col3.metric("极点经度", f"{pole_lon}°")
col4.metric("中央经线", f"{central_m}°")

with st.expander("投影算法说明"):
    st.markdown(textwrap.dedent("""\
    **广义等差分纬线多圆锥投影法** (郝晓光, 2001)

    1. 选定赤道上一点 **P'** (极点经度) 作为新坐标系的正极
    2. 将整个地球绕 z 轴旋转使 P' 到达 λ=90°，再绕 y 轴旋转 -90° 使 P' 到达北极
    3. 在旋转后的坐标系上做**等距圆柱投影** (x = 广义经度, y = 广义纬度)
    4. 最终效果：**纬线为纵轴**，南北极不变形，全球陆地关系接近地球仪

    **操作提示**：拖动左侧滑块实时观察地图变化。
    极点设在不同经度会改变地图的中心视角；中央经线控制投影的水平居中位置。
    """))
