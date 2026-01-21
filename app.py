import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Nueva librería para 3D

# --- Configuración General ---
st.set_page_config(page_title="Sistemas Dinámicos", page_icon="🌀", layout="wide")

# --- Barra Lateral de Navegación ---
st.sidebar.title("Explora el CAOS")
opcion = st.sidebar.selectbox(
    "Elige el sistema:",
    ("Mapa Logístico (2D)", "Atractor de Lorenz (3D)", "Atractor de Thomas (3D)", "Conjunto de Mandelbrot", "Referencias")
)

st.sidebar.divider()

# ==========================================
# OPCIÓN 1: MAPA LOGÍSTICO (Tu código anterior mejorado)
# ==========================================
if opcion == "Mapa Logístico (2D)":
    st.title("El Mapa Logístico")
    st.markdown("Visualizando la ruta hacia el caos en: $x_{n+1} = r x_n (1 - x_n)$")
    
    col1, col2 = st.columns([1, 3]) # Dividimos la pantalla
    
    with col1:
        st.info("Controles")
        n_iter = st.slider("Iteraciones", 500, 2000, 1000)
        r_range = st.slider("Rango de r", 2.5, 4.0, (2.5, 4.0)) # Slider doble
        
    with col2:
        # Lógica de cálculo
        r = np.linspace(r_range[0], r_range[1], 1000)
        x = 1e-5 * np.ones(1000)
        
        # Pre-calentamiento
        for i in range(100): 
            x = r * x * (1 - x)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        # Ciclo principal
        for i in range(n_iter):
            x = r * x * (1 - x)
            ax.scatter(r, x, s=0.1, c='cyan', alpha=0.1)
            
        ax.axis('off') # Quitamos ejes para hacerlo más artístico/minimalista
        st.pyplot(fig)

# ==========================================
# OPCIÓN 2: ATRACTOR DE LORENZ (Nuevo código 3D)
# ==========================================
elif opcion == "Atractor de Lorenz (3D)":
    st.title("Atractor de Lorenz")
    st.markdown(r"""
    El sistema clásico de ecuaciones diferenciales:
    $$
    \begin{cases}
    \frac{dx}{dt} = \sigma(y-x) \\
    \frac{dy}{dt} = x(\rho-z)-y \\
    \frac{dz}{dt} = xy - \beta z
    \end{cases}
    $$
    """)

    # Parámetros en el sidebar específicos para Lorenz
    st.sidebar.header("Parámetros de Lorenz")
    sigma = st.sidebar.slider("Sigma (σ)", 0.0, 20.0, 10.0)
    rho = st.sidebar.slider("Rho (ρ)", 0.0, 50.0, 28.0)
    beta = st.sidebar.slider("Beta (β)", 0.0, 5.0, 2.66)

        # NUEVO: Selector de Paleta de Color
    paleta = st.sidebar.selectbox(
        "Paleta de Color",
        ("Ice", "Plasma", "Viridis", "Inferno", "Turbo", "Twilight")
    )

    # Cálculo de la trayectoria (Método de Euler simple)
    dt = 0.01
    num_steps = 10000
    
    xs, ys, zs = np.empty(num_steps), np.empty(num_steps), np.empty(num_steps)
    xs[0], ys[0], zs[0] = (0.1, 1.0, 1.05) # Punto inicial

    for i in range(num_steps - 1):
        xs[i+1] = xs[i] + (sigma * (ys[i] - xs[i])) * dt
        ys[i+1] = ys[i] + (xs[i] * (rho - zs[i]) - ys[i]) * dt
        zs[i+1] = zs[i] + (xs[i] * ys[i] - beta * zs[i]) * dt

    # Visualización Interactiva con Plotly
    fig = go.Figure(data=go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=zs, colorscale= paleta, width=2), # El color depende de la altura Z
        opacity=0.8
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#0E1117'
        ),
        paper_bgcolor='#0E1117',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# OPCIÓN 3: ATRACTOR DE THOMAS (Con selector de color)
# ==========================================
elif opcion == "Atractor de Thomas (3D)":
    st.title("Atractor de Thomas")
    st.markdown("Un atractor cíclicamente simétrico generado por ecuaciones senoidales. Las ecuaciones diferenciales que describen este sistema son:")
    st.latex(r"""
    \begin{cases}
    \dot{x} = \sin(y) - b x \\
    \dot{y} = \sin(z) - b y \\
    \dot{z} = \sin(x) - b z
    \end{cases}
    """)

    # --- Sidebar ---
    st.sidebar.header("Parámetros Thomas")
    b = st.sidebar.slider("Beta (b)", 0.0, 1.0, 0.205, step=0.001)
    n_steps = st.sidebar.slider("Número de puntos", 10000, 50000, 25000)
    
    # NUEVO: Selector de Paleta de Color
    paleta = st.sidebar.selectbox(
        "Paleta de Color",
        ("Ice", "Plasma", "Viridis", "Inferno", "Turbo", "Twilight")
    )

    # --- Cálculo (Euler) ---
    dt = 0.05
    xs, ys, zs = np.empty(n_steps), np.empty(n_steps), np.empty(n_steps)
    xs[0], ys[0], zs[0] = (0.1, 0, 0) 

    for i in range(n_steps - 1):
        xs[i+1] = xs[i] + (np.sin(ys[i]) - b * xs[i]) * dt
        ys[i+1] = ys[i] + (np.sin(zs[i]) - b * ys[i]) * dt
        zs[i+1] = zs[i] + (np.sin(xs[i]) - b * zs[i]) * dt

    # --- Visualización ---
    fig = go.Figure(data=go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(
            color=xs+ys+zs,  # El color cambia según la posición en el espacio
            colorscale=paleta, # <--- AQUÍ USAMOS LA SELECCIÓN DEL USUARIO
            width=1.5
        ),
        opacity=0.6
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#0E1117'
        ),
        paper_bgcolor='#0E1117',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    # ==========================================
# OPCIÓN 4: CONJUNTO DE MANDELBROT
# ==========================================
elif opcion == "Conjunto de Mandelbrot":
    st.title("El Conjunto de Mandelbrot")
    st.markdown("El fractal más famoso. La frontera del conjunto es infinitamente compleja.")
    st.latex(r"z_{n+1} = z_n^2 + c")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.write("### Parámetros")
        # Menor resolución para que sea rápido en la web
        resolucion = st.slider("Resolución (px)", 200, 1000, 500) 
        iteraciones = st.slider("Profundidad (Iteraciones)", 20, 200, 50)
        
        st.info("Nota: A mayor resolución, más tardará en generarse.")

    with col2:
        # Función optimizada con NumPy (Vectorización)
        def mandelbrot(h, w, max_iter):
            # Crear una rejilla de números complejos
            y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
            c = x + y*1j
            z = c
            divtime = max_iter + np.zeros(z.shape, dtype=int)

            for i in range(max_iter):
                z = z**2 + c
                diverge = z*np.conj(z) > 2**2            
                div_now = diverge & (divtime == max_iter)  
                divtime[div_now] = i                     
                z[diverge] = 2                           

            return divtime

        with st.spinner('Calculando fractal...'):
            plt.figure(figsize=(10, 10))
            # Calculamos el fractal
            fractal = mandelbrot(resolucion, resolucion, iteraciones)
            
            # Visualización
            plt.imshow(fractal, cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
            plt.axis('off')
            # Truco para quitar bordes blancos
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            st.pyplot(plt)

# ==========================================
# OPCIÓN Final: REFERENCIAS
# ==========================================
elif opcion == "Referencias":
    st.title("📚 Bibliografía y Recursos")
    st.markdown("Si te interesa profundizar en estos temas, aquí tienes los recursos esenciales:")
    
    st.subheader("Libros Clásicos")
    st.markdown("""
    * **"Caos: La creación de una ciencia"** - *James Gleick*. (El libro divulgativo por excelencia).
    * **"Nonlinear Dynamics and Chaos"** - *Steven Strogatz*. (La biblia técnica para estudiantes).
    * **"The Fractal Geometry of Nature"** - *Benoît Mandelbrot*. (El libro original del padre de los fractales).
    """)
    
    st.divider()
    
    st.subheader("Librerías de Python utilizadas")
    st.code("""
    import streamlit as st   # Interfaz Web
    import numpy as np       # Cálculo numérico
    import matplotlib.pyplot # Gráficos 2D
    import plotly            # Gráficos 3D
    """)
    
    st.info("Esta web ha sido creada con asistencia de IA y Python.")
