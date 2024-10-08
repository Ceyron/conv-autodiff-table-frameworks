"""
Displays the jaxpr of 1d convolution using `jax.lax.conv_general_dilated` and
its vJp under various configurations.
"""

import streamlit as st
import jax.numpy as jnp
import jax

st.set_page_config(layout="wide")

with st.sidebar:
    what_to_display = st.select_slider(
        "Which vJp to display",
        options=["Input", "Filter", "Both"],
    )
    display_primals = st.checkbox("Display Primals", value=True)
    array_size = st.slider("Array size", 1, 20, 10)
    kernel_size = st.slider("Kernel size", 1, 5, 3)
    batch_size = st.slider("Batch size", 1, 5, 1)
    in_feature_size = st.slider("Input feature size", 1, 5, 1)
    out_feature_size = st.slider("Output feature size", 1, 5, 1)
    if st.toggle("Advanced options"):
        left_padding = st.slider("Left padding", 0, 5, 0)
        right_padding = st.slider("Right padding", 0, 5, 0)
        stride = st.slider("Stride", 1, 4, 1)
        lhs_dilation = st.slider("lhs dilation", 1, 3, 1)
        rhs_dilation = st.slider("rhs dilation", 1, 3, 1)
        if st.toggle("Further options"):
            feature_group_count = st.slider("Feature group count", 1, 5, 1)
            batch_group_count = st.slider("Batch group count", 1, 5, 1)
        else:
            feature_group_count = 1
            batch_group_count = 1
    else:
        left_padding = 0
        right_padding = 0
        stride = 1
        lhs_dilation = 1
        rhs_dilation = 1
        feature_group_count = 1
        batch_group_count = 1

input_array = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_feature_size, array_size))
kernel = jax.random.normal(jax.random.PRNGKey(1), (out_feature_size, in_feature_size, kernel_size))

conv_fun = lambda lhs: jax.lax.conv_general_dilated(
    lhs,
    kernel,
    window_strides=(stride,),
    padding=((left_padding, right_padding),),
    lhs_dilation=(lhs_dilation,),
    rhs_dilation=(rhs_dilation,),
    feature_group_count=feature_group_count,
    batch_group_count=batch_group_count,
)
filter_fun = lambda rhs: jax.lax.conv_general_dilated(
    input_array,
    rhs,
    window_strides=(stride,),
    padding=((left_padding, right_padding),),
    lhs_dilation=(lhs_dilation,),
    rhs_dilation=(rhs_dilation,),
    feature_group_count=feature_group_count,
    batch_group_count=batch_group_count,
)

if what_to_display in ["Input", "Both"]:
    if display_primals:
        conv_fun_jaxpr = jax.make_jaxpr(conv_fun)(input_array)
        st.title("Conv Primal")
        st.code(conv_fun_jaxpr, line_numbers=True)
    out, conv_fun_vjp = jax.vjp(conv_fun, input_array)
    out_cotangent = jax.random.normal(jax.random.PRNGKey(2), out.shape)
    conv_fun_vjp_jaxpr = jax.make_jaxpr(conv_fun_vjp)(out_cotangent)
    st.title("Conv vJp")
    st.code(conv_fun_vjp_jaxpr, line_numbers=True)

if what_to_display in ["Filter", "Both"]:
    if display_primals:
        filter_fun_jaxpr = jax.make_jaxpr(filter_fun)(kernel)
        st.title("Filter Primal")
        st.code(filter_fun_jaxpr, line_numbers=True)
    out, filter_fun_vjp = jax.vjp(filter_fun, kernel)
    out_cotangent = jax.random.normal(jax.random.PRNGKey(2), out.shape)
    filter_fun_vjp_jaxpr = jax.make_jaxpr(filter_fun_vjp)(out_cotangent)
    st.title("Filter vJp")
    st.code(filter_fun_vjp_jaxpr, line_numbers=True)

