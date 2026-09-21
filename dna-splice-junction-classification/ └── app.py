import streamlit as st

from src.predict import predict_sequence


st.set_page_config(
    page_title="DNA Splice-Junction Predictor",
    page_icon="🧬",
    layout="centered"
)


st.title("🧬 DNA Splice-Junction Predictor")

st.write(
    """
    Enter a DNA sequence and the trained deep-learning model
    will predict whether it corresponds to an exon-intron
    boundary, an intron-exon boundary, or neither.
    """
)


sequence = st.text_area(
    "DNA sequence",
    placeholder="Example: ATGCGTACGTTAGCG...",
    height=150
)


if st.button("Analyze sequence"):

    if not sequence.strip():

        st.warning(
            "Please enter a DNA sequence."
        )

    else:

        try:

            result = predict_sequence(
                sequence
            )

            st.success(
                f"Prediction: **{result['class']}**"
            )

            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}"
            )

            st.subheader(
                "Class probabilities"
            )

            for label, probability in result[
                "probabilities"
            ].items():

                st.write(
                    f"**{label}**: {probability:.2%}"
                )

                st.progress(
                    probability
                )

        except ValueError as error:

            st.error(
                str(error)
            )

        except FileNotFoundError:

            st.error(
                "The trained model has not been created yet. "
                "Run the training script first."
            )