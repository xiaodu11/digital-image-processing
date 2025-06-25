import streamlit as st
import cv2
import numpy as np
from PIL import Image


def main():
    st.title("数字图像处理实验")
    st.write("请选择一个实验模块")

    # 创建选项卡
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["图像增强", "边缘检测", "线性变换", "图像锐化", "图像采样与量化", "彩色图像分割"])

    with tab1:
        image_enhancement()

    with tab2:
        edge_detection()

    with tab3:
        linear_transformation()
    with tab4:
        image_sharpening()
    with tab5:
        sampling_and_quantization()
    with tab6:
        color_image_segmentation()


def image_enhancement():
    st.header("图像增强")
    st.write("使用一阶和二阶微分算子进行图像边缘检测和增强")

    # 上传图像
    uploaded_file = st.file_uploader("上传一张图像", type=["jpg", "jpeg", "png"], key="file_uploader_enhancement")

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)
        image = np.array(image)

        # 显示原始图像
        st.image(image, caption="原始图像", use_container_width=True)

        # 选择微分算子
        operator = st.selectbox(
            "选择微分算子",
            ("Sobel (一阶)", "Prewitt (一阶)", "Roberts (一阶)", "Laplacian (二阶)"),
            key="selectbox_enhancement"
        )

        # 应用微分算子
        if st.button("处理图像", key="button_enhancement"):
            processed_image = apply_operator(image, operator)

            # 显示处理后的图像
            st.image(processed_image, caption=f"使用{operator}处理后的图像", use_container_width=True)


def apply_operator(image, operator):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if operator == "Sobel (一阶)":
        # Sobel 算子
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        processed = cv2.magnitude(sobelx, sobely).astype(np.uint8)

    elif operator == "Prewitt (一阶)":
        # Prewitt 算子
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewittx = cv2.filter2D(gray.astype(np.float32), -1, kernelx)
        prewitty = cv2.filter2D(gray.astype(np.float32), -1, kernely)
        processed = cv2.magnitude(prewittx, prewitty).astype(np.uint8)

    elif operator == "Roberts (一阶)":
        # Roberts 算子
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(gray.astype(np.float32), -1, kernelx)
        robertsy = cv2.filter2D(gray.astype(np.float32), -1, kernely)
        processed = cv2.magnitude(robertsx, robertsy).astype(np.uint8)

    elif operator == "Laplacian (二阶)":
        # Laplacian 算子
        processed = cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)

    # 将处理后的图像转换回 BGR 格式以便显示
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    return processed


def edge_detection():
    st.header("边缘检测")
    st.write("使用不同的微分算子进行图像边缘检测")

    # 上传图像
    uploaded_file = st.file_uploader("上传一张图像", type=["jpg", "jpeg", "png"], key="file_uploader_edge")

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)
        image = np.array(image)

        # 显示原始图像
        st.image(image, caption="原始图像", use_container_width=True)

        # 选择微分算子
        operator = st.selectbox(
            "选择微分算子",
            ("Roberts", "Sobel", "Prewitt", "Laplacian", "LoG"),
            key="selectbox_edge"
        )

        # 应用边缘检测
        if st.button("检测边缘", key="button_edge"):
            edge_image = apply_edge_detection(image, operator)

            # 显示边缘检测后的图像
            st.image(edge_image, caption=f"使用{operator}检测到的边缘", use_container_width=True)


def apply_edge_detection(image, operator):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if operator == "Roberts":
        # Roberts 算子
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(gray.astype(np.float32), -1, kernelx)
        robertsy = cv2.filter2D(gray.astype(np.float32), -1, kernely)
        edge = cv2.magnitude(robertsx, robertsy).astype(np.uint8)

    elif operator == "Sobel":
        # Sobel 算子
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = cv2.magnitude(sobelx, sobely).astype(np.uint8)

    elif operator == "Prewitt":
        # Prewitt 算子
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewittx = cv2.filter2D(gray.astype(np.float32), -1, kernelx)
        prewitty = cv2.filter2D(gray.astype(np.float32), -1, kernely)
        edge = cv2.magnitude(prewittx, prewitty).astype(np.uint8)

    elif operator == "Laplacian":
        # Laplacian 算子
        edge = cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)

    elif operator == "LoG":
        # LoG (Laplacian of Gaussian) 算子
        # 先进行高斯模糊，再应用 Laplacian
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Laplacian(blurred, cv2.CV_64F).astype(np.uint8)

    # 将边缘检测后的图像转换回 BGR 格式以便显示
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    return edge


def linear_transformation():
    st.header("线性变换")
    st.write("图像平滑处理和分段线性变换")

    # 上传图像
    uploaded_file = st.file_uploader("上传一张图像", type=["jpg", "jpeg", "png"], key="file_uploader_linear")

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)
        image = np.array(image)

        # 显示原始图像
        st.image(image, caption="原始图像", use_container_width=True)

        # 选择处理类型
        processing_type = st.selectbox(
            "选择处理类型",
            ("平滑处理", "分段线性变换"),
            key="selectbox_processing"
        )

        if processing_type == "平滑处理":
            # 选择滤波器类型
            filter_type = st.selectbox(
                "选择滤波器类型",
                ("中值滤波", "均值滤波"),
                key="selectbox_filter"
            )

            # 选择滤波核大小
            kernel_size = st.slider(
                "选择滤波核大小",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="slider_kernel"
            )

            # 确保中值滤波的核大小是奇数
            if filter_type == "中值滤波" and kernel_size % 2 == 0:
                kernel_size += 1  # 调整为最近的奇数

            # 应用滤波
            if st.button("应用滤波", key="button_filter"):
                filtered_image = apply_filter(image, filter_type, kernel_size)

                # 显示滤波后的图像
                st.image(filtered_image, caption=f"使用{filter_type}处理后的图像", use_container_width=True)

        elif processing_type == "分段线性变换":
            # 选择分段线性变换参数
            a = st.slider("a", 0.0, 1.0, 0.3, 0.01)
            b = st.slider("b", 0.0, 1.0, 0.6, 0.01)
            c = st.slider("c", 0.0, 1.0, 0.1, 0.01)
            d = st.slider("d", 0.0, 1.0, 0.9, 0.01)

            # 应用分段线性变换
            if st.button("应用分段线性变换", key="button_segment"):
                transformed_image = apply_piecewise_linear_transformation(image, a, b, c, d)

                # 显示变换后的图像
                st.image(transformed_image, caption="分段线性变换处理后的图像", use_container_width=True)


def apply_filter(image, filter_type, kernel_size):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if filter_type == "中值滤波":
        # 中值滤波
        filtered = cv2.medianBlur(gray, kernel_size)
    elif filter_type == "均值滤波":
        # 均值滤波
        filtered = cv2.blur(gray, (kernel_size, kernel_size))

    # 将滤波后的图像转换回 BGR 格式以便显示
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    return filtered


def apply_piecewise_linear_transformation(image, a, b, c, d):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将图像归一化到 [0, 1]
    gray_normalized = gray.astype(np.float32) / 255.0

    # 分段线性变换
    transformed = np.zeros_like(gray_normalized)
    transformed[gray_normalized < a] = gray_normalized[gray_normalized < a] * (b / a)
    transformed[(gray_normalized >= a) & (gray_normalized < c)] = gray_normalized[(gray_normalized >= a) & (
            gray_normalized < c)] * ((d - b) / (c - a)) + b
    transformed[gray_normalized >= c] = gray_normalized[gray_normalized >= c] * ((1 - d) / (1 - c)) + d

    # 将图像恢复到 [0, 255]
    transformed = (transformed * 255).astype(np.uint8)

    # 将变换后的图像转换回 BGR 格式以便显示
    transformed = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

    return transformed


def image_sharpening():
    st.header("图像锐化")
    st.write("使用微分算子进行图像锐化")

    # 上传图像
    uploaded_file = st.file_uploader("上传一张图像", type=["jpg", "jpeg", "png"], key="file_uploader_sharpening")

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)
        image = np.array(image)

        # 显示原始图像
        st.image(image, caption="原始图像", use_container_width=True)

        # 选择微分算子
        operator = st.selectbox(
            "选择微分算子",
            ("Sobel", "Prewitt", "Roberts", "Laplacian"),
            key="selectbox_sharpening"
        )

        # 选择方向
        direction = st.selectbox(
            "选择方向",
            ("x", "y"),
            key="selectbox_direction"
        )

        # 应用图像锐化
        if st.button("应用锐化", key="button_sharpening"):
            sharpened_image = apply_sharpening(image, operator, direction)

            # 显示锐化后的图像
            st.image(sharpened_image, caption=f"使用{operator}在{direction}方向锐化后的图像", use_container_width=True)


def apply_sharpening(image, operator, direction):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将原始图像转换为浮点类型
    gray_float = gray.astype(np.float32) / 255.0

    if operator == "Sobel":
        # Sobel 算子
        if direction == "x":
            sobel = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        elif direction == "y":
            sobel = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
        # 将原始图像与微分图像相加进行锐化
        sharpened = cv2.addWeighted(gray_float, 1, cv2.magnitude(sobel, sobel), 1, 0)

    elif operator == "Prewitt":
        # Prewitt 算子
        if direction == "x":
            kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        elif direction == "y":
            kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt = cv2.filter2D(gray_float, -1, kernel)
        # 将原始图像与微分图像相加进行锐化
        sharpened = cv2.addWeighted(gray_float, 1, prewitt, 1, 0)

    elif operator == "Roberts":
        # Roberts 算子
        if direction == "x":
            kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
        elif direction == "y":
            kernel = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        roberts = cv2.filter2D(gray_float, -1, kernel)
        # 将原始图像与微分图像相加进行锐化
        sharpened = cv2.addWeighted(gray_float, 1, roberts, 1, 0)

    elif operator == "Laplacian":
        # Laplacian 算子
        laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)
        # 将原始图像与微分图像相加进行锐化
        sharpened = cv2.addWeighted(gray_float, 1, laplacian, 1, 0)

    # 将锐化后的图像转换回 uint8 格式以便显示
    sharpened = (sharpened * 255).astype(np.uint8)

    # 将锐化后的图像转换回 BGR 格式以便显示
    sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return sharpened


def sampling_and_quantization():
    st.header("图像采样与量化")
    st.write("图像采样和量化处理")

    # 上传图像
    uploaded_file = st.file_uploader("上传一张图像", type=["jpg", "jpeg", "png"], key="file_uploader_saq")

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)
        image = np.array(image)

        # 显示原始图像
        st.image(image, caption="原始图像", use_container_width=True)

        # 选择处理类型
        processing_type = st.selectbox(
            "选择处理类型",
            ("图像采样", "图像量化"),
            key="selectbox_saq"
        )

        if processing_type == "图像采样":
            # 选择采样分辨率
            sample_ratio = st.slider(
                "选择采样分辨率",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="slider_sample"
            )

            # 应用图像采样
            if st.button("应用采样", key="button_sample"):
                sampled_image = apply_sampling(image, sample_ratio)

                # 显示采样后的图像
                st.image(sampled_image, caption=f"采样后的图像 (分辨率: 1/{sample_ratio})", use_container_width=True)

        elif processing_type == "图像量化":
            # 选择量化等级
            quantization_level = st.slider(
                "选择量化等级",
                min_value=2,
                max_value=256,
                value=64,
                step=1,
                key="slider_quantization"
            )

            # 应用图像量化
            if st.button("应用量化", key="button_quantization"):
                quantized_image = apply_quantization(image, quantization_level)

                # 显示量化后的图像
                st.image(quantized_image, caption=f"量化后的图像 (灰度级: {quantization_level})",
                         use_container_width=True)


def apply_sampling(image, sample_ratio):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取原始图像尺寸
    height, width = gray.shape

    # 计算采样后的图像尺寸
    sampled_height = height // sample_ratio
    sampled_width = width // sample_ratio

    # 采样后的图像
    sampled = cv2.resize(gray, (sampled_width, sampled_height), interpolation=cv2.INTER_NEAREST)

    # 将采样后的图像转换回 BGR 格式以便显示
    sampled = cv2.cvtColor(sampled, cv2.COLOR_GRAY2BGR)

    return sampled


def apply_quantization(image, quantization_level):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 量化
    quantized = np.uint8(np.floor(gray / (256 / quantization_level)) * (256 / quantization_level))

    # 将量化后的图像转换回 BGR 格式以便显示
    quantized = cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)

    return quantized


def color_image_segmentation():
    st.header("彩色图像分割")
    st.write("彩色图像分割处理")

    # 上传图像
    uploaded_file = st.file_uploader("上传一张图像", type=["jpg", "jpeg", "png"], key="file_uploader_cis")

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)
        image = np.array(image)

        # 显示原始图像
        st.image(image, caption="原始图像", use_container_width=True)

        # 选择颜色空间
        color_space = st.selectbox(
            "选择颜色空间",
            ("RGB", "HSI"),
            key="selectbox_color_space"
        )

        if color_space == "RGB":
            # RGB颜色空间分割
            if st.button("显示RGB分量", key="button_rgb"):
                r, g, b = cv2.split(image)
                st.image(r, caption="R分量", use_container_width=True)
                st.image(g, caption="G分量", use_container_width=True)
                st.image(b, caption="B分量", use_container_width=True)

            # 选择分割阈值
            lower_thresh = st.slider("选择下界阈值", 0, 255, 50, key="slider_lower_thresh")
            upper_thresh = st.slider("选择上界阈值", 0, 255, 200, key="slider_upper_thresh")

            if st.button("应用RGB分割", key="button_rgb_seg"):
                segmented_image = apply_rgb_segmentation(image, lower_thresh, upper_thresh)
                st.image(segmented_image, caption="分割后的图像", use_container_width=True)

        elif color_space == "HSI":
            # 转换为HSI颜色空间
            hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 显示HSI分量
            if st.button("显示HSI分量", key="button_hsi"):
                h, s, i = cv2.split(hsi_image)
                st.image(h, caption="H分量", use_container_width=True)
                st.image(s, caption="S分量", use_container_width=True)
                st.image(i, caption="I分量", use_container_width=True)

            # 选择分割阈值
            lower_hue = st.slider("选择H下界色调", 0, 179, 20, key="slider_lower_hue")
            upper_hue = st.slider("选择H上界色调", 0, 179, 160, key="slider_upper_hue")
            lower_saturation = st.slider("选择S下界饱和度", 0, 255, 50, key="slider_lower_saturation")
            upper_saturation = st.slider("选择S上界饱和度", 0, 255, 200, key="slider_upper_saturation")
            lower_intensity = st.slider("选择I下界强度", 0, 255, 50, key="slider_lower_intensity")
            upper_intensity = st.slider("选择I上界强度", 0, 255, 200, key="slider_upper_intensity")

            if st.button("应用HSI分割", key="button_hsi_seg"):
                segmented_image = apply_hsi_segmentation(hsi_image, lower_hue, upper_hue, lower_saturation,
                                                         upper_saturation, lower_intensity, upper_intensity)
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
                st.image(segmented_image, caption="分割后的图像", use_container_width=True)


def apply_rgb_segmentation(image, lower_thresh, upper_thresh):
    # 创建一个掩码，其中满足阈值条件的像素为白色，其余为黑色
    lower = np.array([lower_thresh, lower_thresh, lower_thresh])
    upper = np.array([upper_thresh, upper_thresh, upper_thresh])
    mask = cv2.inRange(image, lower, upper)

    # 应用掩码到原始图像
    segmented = cv2.bitwise_and(image, image, mask=mask)

    return segmented


def apply_hsi_segmentation(hsi_image, lower_hue, upper_hue, lower_saturation, upper_saturation, lower_intensity,
                           upper_intensity):
    # 创建一个掩码，其中满足阈值条件的像素为白色，其余为黑色
    lower = np.array([lower_hue, lower_saturation, lower_intensity])
    upper = np.array([upper_hue, upper_saturation, upper_intensity])
    mask = cv2.inRange(hsi_image, lower, upper)

    # 应用掩码到HSI图像
    segmented = cv2.bitwise_and(hsi_image, hsi_image, mask=mask)

    return segmented


if __name__ == "__main__":
    main()
