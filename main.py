import re
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Biochemical Reference Data ---
AA_PROPERTIES = {
    "A": {"name": "Alanine", "charge": 0, "hydro": 1.8, "mw": 89.1, "group": "Nonpolar"},
    "R": {"name": "Arginine", "charge": 1, "hydro": -4.5, "mw": 174.2, "group": "Positive"},
    "N": {"name": "Asparagine", "charge": 0, "hydro": -3.5, "mw": 132.1, "group": "Polar"},
    "D": {"name": "Aspartate", "charge": -1, "hydro": -3.5, "mw": 133.1, "group": "Negative"},
    "C": {"name": "Cysteine", "charge": 0, "hydro": 2.5, "mw": 121.2, "group": "Special"},
    "Q": {"name": "Glutamine", "charge": 0, "hydro": -3.5, "mw": 146.2, "group": "Polar"},
    "E": {"name": "Glutamate", "charge": -1, "hydro": -3.5, "mw": 147.1, "group": "Negative"},
    "G": {"name": "Glycine", "charge": 0, "hydro": -0.4, "mw": 75.1, "group": "Special"},
    "H": {"name": "Histidine", "charge": 0.1, "hydro": -3.2, "mw": 155.2, "group": "Positive"},
    "I": {"name": "Isoleucine", "charge": 0, "hydro": 4.5, "mw": 131.2, "group": "Nonpolar"},
    "L": {"name": "Leucine", "charge": 0, "hydro": 3.8, "mw": 131.2, "group": "Nonpolar"},
    "K": {"name": "Lysine", "charge": 1, "hydro": -3.9, "mw": 146.2, "group": "Positive"},
    "M": {"name": "Methionine", "charge": 0, "hydro": 1.9, "mw": 149.2, "group": "Nonpolar"},
    "F": {"name": "Phenylalanine", "charge": 0, "hydro": 2.8, "mw": 165.2, "group": "Aromatic"},
    "P": {"name": "Proline", "charge": 0, "hydro": -1.6, "mw": 115.1, "group": "Special"},
    "S": {"name": "Serine", "charge": 0, "hydro": -0.8, "mw": 105.1, "group": "Polar"},
    "T": {"name": "Threonine", "charge": 0, "hydro": -0.7, "mw": 119.1, "group": "Polar"},
    "W": {"name": "Tryptophan", "charge": 0, "hydro": -0.9, "mw": 204.2, "group": "Aromatic"},
    "Y": {"name": "Tyrosine", "charge": 0, "hydro": -1.3, "mw": 181.2, "group": "Aromatic"},
    "V": {"name": "Valine", "charge": 0, "hydro": 4.2, "mw": 117.1, "group": "Nonpolar"},
}

RELEVANT_FEATURES = {
    "Domain",
    "Region",
    "Active site",
    "Binding site",
    "DNA binding",
    "Zinc finger",
    "Transmembrane",
    "Motif",
}


#Backend i think
@st.cache_data(ttl=3600)
def fetch_uniprot_data(accession: str) -> dict:
    """Fetch protein metadata, sequence, and features from UniProt REST API."""
    accession = accession.strip().upper()
    if not accession:
        raise ValueError("Please enter a UniProt code ID.")
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Could not find UniProt ID '{accession}' (HTTP {response.status_code}).")

    data = response.json()
    protein_name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "Unknown Protein")
    )
    organism = data.get("organism", {}).get("scientificName", "Unknown Organism")
    sequence = data.get("sequence", {}).get("value", "")

    features = []
    for feat in data.get("features", []):
        ftype = feat.get("type")
        if ftype in RELEVANT_FEATURES:
            start = feat.get("location", {}).get("start", {}).get("value")
            end = feat.get("location", {}).get("end", {}).get("value")
            desc = feat.get("description", ftype)
            if start and end:
                features.append({
                    "type": ftype,
                    "description": desc,
                    "start": int(start),
                    "end": int(end),
                })

    return {
        "accession": accession.strip().upper(),
        "name": protein_name,
        "organism": organism,
        "sequence": sequence,
        "length": len(sequence),
        "features": features,
    }


def analyze_variants(raw_variants: str, protein: dict) -> tuple[pd.DataFrame, list[str]]:
    """Parse point mutations, validate against wild-type sequence, and score shifts."""
    tokens = [t.strip().upper() for t in re.split(r"[\s,;]+", raw_variants) if t.strip()]
    sequence = protein["sequence"]
    features = protein["features"]

    results = []
    errors = []

    for token in tokens:
        match = re.match(r"^([A-Z])(\d+)([A-Z])$", token)
        if not match:
            errors.append(f"`{token}`: Invalid format. Expected format like `R175H`.")
            continue

        wt_aa, pos_str, mut_aa = match.groups()
        pos = int(pos_str)

        if wt_aa not in AA_PROPERTIES or mut_aa not in AA_PROPERTIES:
            errors.append(f"`{token}`: Unknown amino acid code.")
            continue

        if pos < 1 or pos > len(sequence):
            errors.append(f"`{token}`: Position {pos} is out of bounds (1–{len(sequence)}).")
            continue

        actual_aa = sequence[pos - 1]
        if actual_aa != wt_aa:
            errors.append(
                f"`{token}`: Wild-type mismatch at position {pos}. "
                f"Expected `{actual_aa}`, got `{wt_aa}`."
            )
            continue

        wt_props = AA_PROPERTIES[wt_aa]
        mut_props = AA_PROPERTIES[mut_aa]

        charge_delta = round(mut_props["charge"] - wt_props["charge"], 1)
        hydro_delta = round(mut_props["hydro"] - wt_props["hydro"], 2)
        mw_delta = round(mut_props["mw"] - wt_props["mw"], 1)

        overlapping = [
            f"{f['type']}: {f['description']} ({f['start']}-{f['end']})"
            for f in features
            if f["start"] <= pos <= f["end"]
        ]

        #impct score
        impact_score = 0
        if wt_props["group"] != mut_props["group"]:
            impact_score += 1
        if abs(charge_delta) >= 1:
            impact_score += 2
        if abs(hydro_delta) >= 2.5:
            impact_score += 1
        if overlapping:
            impact_score += 2

        if impact_score >= 4:
            severity = "High"
        elif impact_score >= 2:
            severity = "Moderate"
        else:
            severity = "Low"

        results.append({
            "Variant": token,
            "Position": pos,
            "Wild-Type": f"{wt_aa} ({wt_props['group']})",
            "Mutant": f"{mut_aa} ({mut_props['group']})",
            "Charge Shift": f"{charge_delta:+g}",
            "Hydrophobicity Shift": f"{hydro_delta:+.2f}",
            "MW Shift (Da)": f"{mw_delta:+.1f}",
            "Overlapping Regions": "; ".join(overlapping) if overlapping else "None",
            "Predicted Disruption": severity,
        })

    return pd.DataFrame(results), errors


def plot_protein_map(protein: dict, variants_df: pd.DataFrame) -> go.Figure:
    """Linear domain and mutation."""
    fig = go.Figure()
    seq_len = protein["length"]

    #protein
    fig.add_trace(go.Scatter(
        x=[1, seq_len],
        y=[0, 0],
        mode="lines",
        line=dict(color="#CBD5E1", width=10),
        name="Protein Backbone",
        hoverinfo="skip",
    ))

    #proten feature
    palette = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#06B6D4"]
    for idx, feat in enumerate(protein["features"]):
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=[feat["start"], feat["end"]],
            y=[0, 0],
            mode="lines",
            line=dict(color=color, width=18),
            name=f"{feat['type']}: {feat['description'][:28]}",
            hovertemplate=(
                f"<b>{feat['type']}</b><br>"
                f"{feat['description']}<br>"
                f"Residues: {feat['start']}–{feat['end']}<extra></extra>"
            ),
        ))

    #mutate marker
    if not variants_df.empty:
        color_map = {"High": "#EF4444", "Moderate": "#F97316", "Low": "#22C55E"}
        for _, row in variants_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Position"]],
                y=[0.22],
                mode="markers+text",
                marker=dict(
                    size=14,
                    symbol="triangle-down",
                    color=color_map.get(row["Predicted Disruption"], "#EF4444"),
                ),
                text=[row["Variant"]],
                textposition="top center",
                name=row["Variant"],
                hovertemplate=(
                    f"<b>{row['Variant']}</b> (Pos {row['Position']})<br>"
                    f"Disruption: {row['Predicted Disruption']}<br>"
                    f"Region: {row['Overlapping Regions']}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=f"Linear Domain & Mutation Map ({protein['accession']})",
        xaxis=dict(title="Amino Acid Position", range=[0, seq_len + 10], showgrid=False),
        yaxis=dict(range=[-0.4, 0.7], visible=False),
        height=280,
        margin=dict(l=20, r=20, t=45, b=35),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="left", x=0),
    )
    return fig


# UI
st.set_page_config(page_title="Protein Variant Explorer", page_icon="🧬", layout="wide")

st.title(" Protein Variant and  Mutation ")
st.markdown(
    "Analyze Protein from Uniprot "
)

with st.sidebar:
    st.header("Input Parameters")
    uniprot_id = st.text_input("UniProt Accession ID", value="P04637", help="Example: P04637 (TP53), P00533 (EGFR)")
    variants_input = st.text_area(
        "Missense Variants (comma or newline separated)",
        value="R175H\nG245S\nR248Q\nR273H\nP72R",
        height=140,
    )
    run_btn = st.button("Analyze Variants", type="primary", use_container_width=True)

    st.divider()
    st.caption("Quick Presets:")
    st.markdown("- **TP53 (`P04637`):** `R175H, G245S, R248Q, P72R`")
    st.markdown("- **BRAF (`P15056`):** `V600E, G469A, K601E`")
    st.markdown("- **HBB (`P68871`):** `E7V, E7K`")

if run_btn or uniprot_id:
    try:
        with st.spinner(f"Fetching {uniprot_id.upper()} from UniProt..."):
            protein = fetch_uniprot_data(uniprot_id)

        col1, col2, col3 = st.columns(3)
        col1.metric("Protein", protein["name"])
        col2.metric("Organism", protein["organism"])
        col3.metric("Sequence Length", f"{protein['length']} aa")

        variants_df, errors = analyze_variants(variants_input, protein)
        for err in errors:
            st.warning(err)

        # Interactive Sequence & Domain Map
        st.plotly_chart(plot_protein_map(protein, variants_df), use_container_width=True)

        # Variant Analysis Table
        if not variants_df.empty:
            st.subheader("Variant Biochemical & Domain Impact")
            st.dataframe(variants_df, use_container_width=True, hide_index=True)
            csv_data = variants_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=f"{protein['accession']}_variant_report.csv",
                mime="text/csv",
            )

        # Expander for full sequence & domain table
        with st.expander("View Annotated UniProt Domains & Reference Sequence"):
            if protein["features"]:
                st.dataframe(
                    pd.DataFrame(protein["features"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No domain annotations returned for this entry.")
            st.code(protein["sequence"], language="text")
    except Exception as exc:
        st.error(str(exc))
