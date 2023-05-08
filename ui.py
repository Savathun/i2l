import streamlit as st
from PIL import Image

import ui_backend


@st.cache_resource
def get_backend(): return ui_backend.Image2Latex()


if __name__ == '__main__':
    st.set_page_config(page_title='I2L')
    st.title('Image to LaTeX')
    st.markdown('Convert images to LaTeX code.')
    uploaded_file = st.file_uploader('Upload an image of an equation', type=['png', 'jpg'])
    image2latex = get_backend()
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

    if st.button('Convert'):
        if uploaded_file and image:
            with st.spinner('Computing'):
                latex_code = image2latex.predict(image)
                st.code(latex_code, language='latex')
                st.markdown(f'$\\displaystyle {latex_code}$')
        else:
            st.error('Please upload an image.')
