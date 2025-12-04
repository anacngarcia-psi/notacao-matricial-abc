import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# Helper functions
# -----------------------------

def parse_list_cell(cell):
    """Parse a comma-separated cell into a clean list of strings."""
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]

def build_matrices_from_df(df):
    """
    From a DataFrame with columns:
        - 'Experience'
        - 'Symbols_list'
        - 'Emotions_list'
    build A (experiences x symbols), B (experiences x affects),
    and the symbol/affect vocabularies.
    """
    # Collect vocabularies
    all_symbols = sorted({s for row in df["Symbols_list"] for s in row})
    all_affects = sorted({e for row in df["Emotions_list"] for e in row})

    symbols = list(all_symbols)
    affects = list(all_affects)

    n_exp = len(df)
    n_sym = len(symbols)
    n_aff = len(affects)

    # Build A: experiences × symbols
    A = np.zeros((n_exp, n_sym), dtype=float)
    sym_index = {s: j for j, s in enumerate(symbols)}
    for i, row_syms in enumerate(df["Symbols_list"]):
        for s in row_syms:
            j = sym_index[s]
            A[i, j] += 1.0  # counts; change to =1.0 for pure incidence

    # Build B: experiences × affects
    B = np.zeros((n_exp, n_aff), dtype=float)
    aff_index = {e: j for j, e in enumerate(affects)}
    for i, row_emos in enumerate(df["Emotions_list"]):
        for e in row_emos:
            j = aff_index[e]
            B[i, j] += 1.0

    return A, B, symbols, affects

def svd_decompose_C(C):
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return U, S, Vt

def build_rank1_mode(U, S, Vt):
    """Return C1 (rank-1), s1, u1, v1."""
    if len(S) == 0:
        return None, 0.0, None, None
    s1 = S[0]
    u1 = U[:, 0]
    v1 = Vt[0, :]
    C1 = s1 * np.outer(u1, v1)
    return C1, s1, u1, v1

def dominant_pair(C, row_labels, col_labels):
    """Get the (row_label, col_label, value) of the dominant cell in |C|."""
    if C.size == 0:
        return None, None, 0.0
    idx = np.unravel_index(np.abs(C).argmax(), C.shape)
    i, j = idx
    return row_labels[i], col_labels[j], C[i, j]

def energy_shares(S):
    """Energy shares of singular values (s_i^2 / sum s_j^2)."""
    if len(S) == 0:
        return np.array([])
    s2 = S ** 2
    total = s2.sum()
    if total == 0:
        return np.zeros_like(S)
    return s2 / total

def compute_latent_positions(U, S, Vt, n_dims=2):
    """
    Compute latent coordinates for symbols and affects using first n_dims.
    For simplicity, we use U[:, :n_dims], Vt.T[:, :n_dims].
    """
    if len(S) == 0:
        return None, None

    k = min(n_dims, len(S))
    sym_coords = U[:, :k]      # symbols in latent space
    aff_coords = Vt.T[:, :k]   # affects in latent space
    return sym_coords, aff_coords

def deflate_once(C, symbols, affects, alpha=1.0):
    """
    Removes alpha * C1 from C.
    Returns:
      - C_resid: residual matrix C - alpha * C1
      - C1: the removed rank-1 component
      - info: same dict as dominant_mode
    """
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    
    if len(S) == 0:
        return C, None, None
    
    s1 = S[0]
    u1 = U[:, 0]
    v1 = Vt[0, :]

    # rank-1 reconstruction
    C1 = s1 * np.outer(u1, v1)

    # where C and C1 are strongest
    jC, kC = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    j1, k1 = np.unravel_index(np.argmax(np.abs(C1)), C.shape)

    # energy / explanation
    total_energy = np.sum(S**2)
    energy_1 = s1**2
    frac_energy = energy_1 / (total_energy + 1e-12)

    info = {
        "s1": s1,
        "dominant_symbol_in_C": symbols[jC],
        "dominant_affect_in_C": affects[kC],
        "dominant_value_in_C": C[jC, kC],
        "dominant_symbol_in_C1": symbols[j1],
        "dominant_affect_in_C1": affects[k1],
        "dominant_value_in_C1": C1[j1, k1],
        "energy_total": total_energy,
        "energy_mode1": energy_1,
        "energy_share_mode1": frac_energy,
    }

    C_resid = C - alpha * C1
    return C_resid, C1, info

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="Ferramenta de Notação Matricial", layout="wide")

st.title("Ferramenta de Notação Matricial")
st.markdown(
    """
Esta aplicação implementa o modelo A/B/C:

- Experiências são as linhas.
- Símbolos e afetos (emoções) constroem duas matrizes de incidência **A** e **B**.
- Calculamos **C = Aᵀ B**, e então realizamos SVD para extrair o modo
  simbólico-afetivo dominante (a *idéia máxima*).
"""
)

# --- Data input ---

st.header("1. Insira experiências, símbolos e afetos")

st.markdown(
    """
Preencha a tabela abaixo. Para cada experiência:

- **Experiência**: uma breve descrição.
- **Símbolos**: lista separada por vírgulas (ex: `mesa, vinho, silêncio`).
- **Emoções**: lista separada por vírgulas (ex: `raiva, tristeza`).
"""
)

example_data = pd.DataFrame(
    [
        {
            "Experiência": "Discussão com parceiro no jantar",
            "Símbolos (separados por vírgula)": "mesa, vinho, silêncio",
            "Emoções (separadas por vírgula)": "raiva, tristeza",
        },
        {
            "Experiência": "Apresentação no trabalho que foi bem",
            "Símbolos (separados por vírgula)": "slides, chefe, aplausos",
            "Emoções (separadas por vírgula)": "orgulho, alívio",
        },
    ]
)

data = st.data_editor(
    example_data,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor",
)

run = st.button("Executar análise", type="primary")

if run:
    # Clean and parse
    df = data.copy()

    # Drop completely empty rows
    df = df[~df["Experiência"].astype(str).str.strip().eq("")]

    if df.empty:
        st.warning("Por favor, insira pelo menos uma experiência.")
        st.stop()

    df["Symbols_list"] = df["Símbolos (separados por vírgula)"].apply(parse_list_cell)
    df["Emotions_list"] = df["Emoções (separadas por vírgula)"].apply(parse_list_cell)

    # Remove rows with no symbols and no emotions
    df = df[(df["Symbols_list"].map(len) > 0) | (df["Emotions_list"].map(len) > 0)]

    if df.empty:
        st.warning("Todas as linhas têm símbolos e emoções vazias — nada para analisar.")
        st.stop()

    st.subheader("Dados processados")
    st.write(df[["Experiência", "Symbols_list", "Emotions_list"]])

    # --- Build A, B, C (silently, without displaying) ---
    A, B, symbols, affects = build_matrices_from_df(df)
    C = A.T @ B  # symbols × affects

    # Compute SVD for later use
    U, S, Vt = svd_decompose_C(C)

    if len(S) == 0:
        st.warning("C está vazia ou é zero; não é possível calcular SVD.")
        st.stop()

    # --- Latent maps (symbols + affects together) ---
    st.header("2. Mapa latente conjunto para símbolos E afetos")

    sym_coords, aff_coords = compute_latent_positions(U, S, Vt, n_dims=2)

    if sym_coords is not None:
        # Symbols
        sym_df = pd.DataFrame(sym_coords[:, :2], columns=["dim1", "dim2"])
        sym_df["rótulo"] = symbols
        sym_df["tipo"] = "símbolo"

        # Affects
        aff_df = pd.DataFrame(aff_coords[:, :2], columns=["dim1", "dim2"])
        aff_df["rótulo"] = affects
        aff_df["tipo"] = "afeto"

        # Combine into one dataframe
        latent_df = pd.concat([sym_df, aff_df], ignore_index=True)

        st.subheader("Espaço latente (símbolos + afetos no mesmo gráfico)")

        import plotly.express as px
        
        fig_latent = px.scatter(
            latent_df,
            x="dim1",
            y="dim2",
            text="rótulo",
            color="tipo",        # different color for symbols vs affects
            symbol="tipo",       # different marker shape as well
            title="Mapa latente de símbolos e afetos",
        )
        fig_latent.update_traces(textposition="top center")
        st.plotly_chart(fig_latent, use_container_width=True)
        
        # Show table with coordinates
        st.subheader("Tabela de coordenadas")
        st.markdown("Esta tabela mostra as coordenadas exatas de cada símbolo e afeto no espaço latente.")
        
        # Create a cleaner display dataframe
        display_df = latent_df.copy()
        display_df = display_df.rename(columns={
            "rótulo": "Símbolo/Afeto",
            "tipo": "Tipo",
            "dim1": "Dimensão 1",
            "dim2": "Dimensão 2"
        })
        display_df = display_df[["Símbolo/Afeto", "Tipo", "Dimensão 1", "Dimensão 2"]]
        
        # Sort by type (símbolos first) then by label
        display_df = display_df.sort_values(["Tipo", "Símbolo/Afeto"])
        
        st.dataframe(
            display_df.style.format({
                "Dimensão 1": "{:.4f}",
                "Dimensão 2": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Valores singulares não-zero insuficientes para construir mapas latentes.")

    # --- Iterative deflation ---
    st.header("3. Deflação iterativa: extraindo todos os modos")

    st.markdown(
        """
Esta seção realiza **deflação iterativa** — extrai e remove repetidamente o modo dominante 
até que a matriz esteja completamente decomosta. Cada iteração revela:
- O valor singular dominante (força do modo)
- A parcela de energia (proporção da variância total explicada)
- O par símbolo-afeto dominante para aquele modo
"""
    )

    # Calculate total energy from original matrix (for percentage calculations)
    total_energy_original = np.sum(S**2)

    C_current = C.copy()
    modes_info = []
    max_modes = min(C.shape[0], C.shape[1], 50)  # Limit to avoid excessive computation

    for k in range(max_modes):
        C_current_next, Ck, info_k = deflate_once(C_current, symbols, affects, alpha=1.0)
        
        if info_k is None or abs(info_k['s1']) < 1e-9:
            break
        
        # Calculate energy share relative to ORIGINAL total energy
        energy_share_relative_to_original = (info_k['s1']**2) / (total_energy_original + 1e-12)
        
        modes_info.append({
            "Modo": k + 1,
            "Valor Singular (s)": info_k['s1'],
            "Parcela de Energia": energy_share_relative_to_original,
            "Símbolo Dominante": info_k["dominant_symbol_in_C1"],
            "Afeto Dominante": info_k["dominant_affect_in_C1"],
            "Valor": info_k["dominant_value_in_C1"]
        })
        
        C_current = C_current_next

    if modes_info:
        modes_df = pd.DataFrame(modes_info)
        
        st.subheader(f"Extraídos {len(modes_info)} modos")
        st.dataframe(
            modes_df.style.format({
                "Valor Singular (s)": "{:.3f}",
                "Parcela de Energia": "{:.3f}",
                "Valor": "{:.3f}"
            }),
            use_container_width=True
        )

        # Cumulative energy plot
        modes_df["Energia Cumulativa"] = modes_df["Parcela de Energia"].cumsum()
        
        try:
            import plotly.graph_objects as go
            
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Bar(
                x=modes_df["Modo"],
                y=modes_df["Parcela de Energia"],
                name="Individual",
                marker_color='lightblue'
            ))
            fig_energy.add_trace(go.Scatter(
                x=modes_df["Modo"],
                y=modes_df["Energia Cumulativa"],
                name="Cumulativa",
                mode='lines+markers',
                marker_color='red',
                yaxis='y2'
            ))
            
            fig_energy.update_layout(
                title="Distribuição de energia entre os modos",
                xaxis_title="Modo",
                yaxis_title="Parcela de Energia (individual)",
                yaxis2=dict(
                    title="Energia Cumulativa",
                    overlaying='y',
                    side='right',
                    range=[0, 1.1]
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_energy, use_container_width=True)
        except Exception:
            st.line_chart(modes_df.set_index("Modo")["Energia Cumulativa"])

        # Summary
        cumulative_top3 = modes_df.head(3)["Parcela de Energia"].sum()
        st.markdown(
            f"""
**Resumo:**
- Total de modos extraídos: **{len(modes_info)}**
- Os 3 primeiros modos explicam: **{cumulative_top3:.1%}** da variância total
- Primeiro modo (idéia máxima) explica: **{modes_df.iloc[0]['Parcela de Energia']:.1%}**
"""
        )
    else:
        st.warning("Nenhum modo significativo encontrado para deflação.")
