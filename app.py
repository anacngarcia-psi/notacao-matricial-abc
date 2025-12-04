import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# Helper functions
# -----------------------------

def parse_list_cell(cell):
    """Parse a comma-separated cell into a clean list of strings (lowercase)."""
    if pd.isna(cell):
        return []
    return [x.strip().lower() for x in str(cell).split(",") if x.strip()]

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
    if C.size == 0:
        return C, None, None
    
    jC, kC = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    j1, k1 = np.unravel_index(np.argmax(np.abs(C1)), C.shape)

    # Validate indices are within bounds
    if jC >= len(symbols) or kC >= len(affects) or j1 >= len(symbols) or k1 >= len(affects):
        # Dimensions mismatch - return None to skip this mode
        return C, None, None

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
        {
            "Experiência": "Briga com irmão por telefone",
            "Símbolos (separados por vírgula)": "Telefone, Gritos, Silêncio",
            "Emoções (separadas por vírgula)": "Raiva, Frustração",
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

        # Find items with same or very close coordinates
        def find_nearby_items(df, threshold=0.01):
            """Find items that are at the same or very close coordinates."""
            nearby_info = []
            for i, row in df.iterrows():
                nearby = []
                for j, other_row in df.iterrows():
                    if i != j:
                        # Calculate euclidean distance
                        dist = np.sqrt((row['dim1'] - other_row['dim1'])**2 + 
                                     (row['dim2'] - other_row['dim2'])**2)
                        if dist < threshold:
                            nearby.append(f"{other_row['rótulo']} ({other_row['tipo']})")
                
                if nearby:
                    nearby_info.append(f"⚠️ Também aqui: {', '.join(nearby)}")
                else:
                    nearby_info.append("✓ Posição única")
            return nearby_info
        
        latent_df['nearby'] = find_nearby_items(latent_df)

        st.subheader("Espaço latente (símbolos + afetos no mesmo gráfico)")
        st.markdown("💡 **Dica:** Passe o mouse sobre os pontos para ver se há outros itens nas mesmas coordenadas!")

        import plotly.express as px
        import plotly.graph_objects as go
        
        # Create scatter plot with custom hover data
        fig_latent = px.scatter(
            latent_df,
            x="dim1",
            y="dim2",
            text="rótulo",
            color="tipo",
            symbol="tipo",
            title="Mapa latente de símbolos e afetos",
            custom_data=['rótulo', 'tipo', 'nearby', 'dim1', 'dim2']
        )
        
        # Customize hover template
        fig_latent.update_traces(
            textposition="top center",
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Tipo: %{customdata[1]}<br>" +
                         "Coordenadas: (%{customdata[3]:.4f}, %{customdata[4]:.4f})<br>" +
                         "%{customdata[2]}<br>" +
                         "<extra></extra>"
        )
        
        st.plotly_chart(fig_latent, use_container_width=True)
        
        # Show overlapping items grouped
        st.subheader("📍 Itens com coordenadas idênticas ou muito próximas")
        st.markdown("Esta seção agrupa automaticamente símbolos e afetos que estão no mesmo lugar ou muito próximos no mapa.")
        
        # Group items by proximity
        def group_by_proximity(df, threshold=0.01):
            """Group items that are close together."""
            groups = []
            processed = set()
            
            for i, row in df.iterrows():
                if i in processed:
                    continue
                    
                # Find all items close to this one
                group = {
                    'coord': (row['dim1'], row['dim2']),
                    'items': [(row['rótulo'], row['tipo'])]
                }
                processed.add(i)
                
                for j, other_row in df.iterrows():
                    if i != j and j not in processed:
                        dist = np.sqrt((row['dim1'] - other_row['dim1'])**2 + 
                                     (row['dim2'] - other_row['dim2'])**2)
                        if dist < threshold:
                            group['items'].append((other_row['rótulo'], other_row['tipo']))
                            processed.add(j)
                
                groups.append(group)
            
            # Sort: groups with multiple items first, then by number of items
            groups.sort(key=lambda x: (-len(x['items']), x['coord']))
            return groups
        
        groups = group_by_proximity(latent_df)
        
        # Display groups with overlaps prominently
        overlapping_groups = [g for g in groups if len(g['items']) > 1]
        single_items = [g for g in groups if len(g['items']) == 1]
        
        if overlapping_groups:
            st.markdown("### ⚠️ Sobreposições detectadas")
            st.markdown(f"**{len(overlapping_groups)} grupo(s)** com múltiplos itens na mesma posição:")
            
            for idx, group in enumerate(overlapping_groups, 1):
                coord = group['coord']
                items = group['items']
                
                # Create expandable section for each group
                with st.expander(f"**Grupo {idx}** — {len(items)} itens sobrepostos", expanded=True):
                    st.markdown(f"**Coordenadas:** `({coord[0]:.4f}, {coord[1]:.4f})`")
                    st.markdown("**Itens nesta posição:**")
                    
                    # Create a nice list of items
                    for item_name, item_type in sorted(items):
                        emoji = "🔵" if item_type == "símbolo" else "🔷"
                        st.markdown(f"{emoji} **{item_name}** ({item_type})")
        else:
            st.success("✅ Nenhuma sobreposição detectada! Todos os itens têm posições únicas no mapa.")
        
        # Show single items in a collapsible section
        if single_items:
            with st.expander(f"✓ Itens com posição única ({len(single_items)} itens)"):
                st.markdown("Estes itens não estão sobrepostos com nenhum outro:")
                
                # Group by type for better organization
                unique_symbols = [(name, coord) for g in single_items for name, tipo in g['items'] 
                          if tipo == "símbolo" for coord in [g['coord']]]
                unique_affects = [(name, coord) for g in single_items for name, tipo in g['items'] 
                          if tipo == "afeto" for coord in [g['coord']]]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if unique_symbols:
                        st.markdown("**Símbolos:**")
                        for name, coord in sorted(unique_symbols):
                            st.markdown(f"• {name} `({coord[0]:.4f}, {coord[1]:.4f})`")
                
                with col2:
                    if unique_affects:
                        st.markdown("**Afetos:**")
                        for name, coord in sorted(unique_affects):
                            st.markdown(f"• {name} `({coord[0]:.4f}, {coord[1]:.4f})`")
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
