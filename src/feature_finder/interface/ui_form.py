# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QGraphicsView, QGridLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QSpinBox,
    QTextEdit, QWidget)

class Ui_featureFinder(object):
    def setupUi(self, featureFinder):
        if not featureFinder.objectName():
            featureFinder.setObjectName(u"featureFinder")
        featureFinder.resize(982, 457)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(featureFinder.sizePolicy().hasHeightForWidth())
        featureFinder.setSizePolicy(sizePolicy)
        featureFinder.setMinimumSize(QSize(0, 0))
        featureFinder.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_2 = QGridLayout(featureFinder)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.stream_window = QGraphicsView(featureFinder)
        self.stream_window.setObjectName(u"stream_window")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.stream_window.sizePolicy().hasHeightForWidth())
        self.stream_window.setSizePolicy(sizePolicy1)
        self.stream_window.setMinimumSize(QSize(471, 411))
        self.stream_window.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.gridLayout_2.addWidget(self.stream_window, 0, 0, 3, 1)

        self.image_import_frame = QFrame(featureFinder)
        self.image_import_frame.setObjectName(u"image_import_frame")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.image_import_frame.sizePolicy().hasHeightForWidth())
        self.image_import_frame.setSizePolicy(sizePolicy2)
        self.image_import_frame.setMaximumSize(QSize(16777215, 125))
        self.image_import_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_import_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_4 = QGridLayout(self.image_import_frame)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.file_path_label = QLabel(self.image_import_frame)
        self.file_path_label.setObjectName(u"file_path_label")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.file_path_label.sizePolicy().hasHeightForWidth())
        self.file_path_label.setSizePolicy(sizePolicy3)
        self.file_path_label.setMinimumSize(QSize(100, 0))
        self.file_path_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_4.addWidget(self.file_path_label, 2, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.save_image_label = QLabel(self.image_import_frame)
        self.save_image_label.setObjectName(u"save_image_label")
        sizePolicy3.setHeightForWidth(self.save_image_label.sizePolicy().hasHeightForWidth())
        self.save_image_label.setSizePolicy(sizePolicy3)
        self.save_image_label.setMinimumSize(QSize(100, 0))
        self.save_image_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_4.addWidget(self.save_image_label, 3, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.save_image_button = QPushButton(self.image_import_frame)
        self.save_image_button.setObjectName(u"save_image_button")
        self.save_image_button.setEnabled(False)
        sizePolicy.setHeightForWidth(self.save_image_button.sizePolicy().hasHeightForWidth())
        self.save_image_button.setSizePolicy(sizePolicy)

        self.gridLayout_4.addWidget(self.save_image_button, 3, 1, 1, 1)

        self.file_path_entry = QTextEdit(self.image_import_frame)
        self.file_path_entry.setObjectName(u"file_path_entry")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.file_path_entry.sizePolicy().hasHeightForWidth())
        self.file_path_entry.setSizePolicy(sizePolicy4)
        self.file_path_entry.setMinimumSize(QSize(250, 0))
        self.file_path_entry.setMaximumSize(QSize(16777215, 35))
        self.file_path_entry.setFrameShape(QFrame.Shape.StyledPanel)
        self.file_path_entry.setFrameShadow(QFrame.Shadow.Plain)
        self.file_path_entry.setLineWidth(1)
        self.file_path_entry.setMidLineWidth(0)
        self.file_path_entry.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.file_path_entry.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.file_path_entry.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        self.gridLayout_4.addWidget(self.file_path_entry, 2, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.file_path_browse_button = QPushButton(self.image_import_frame)
        self.file_path_browse_button.setObjectName(u"file_path_browse_button")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.file_path_browse_button.sizePolicy().hasHeightForWidth())
        self.file_path_browse_button.setSizePolicy(sizePolicy5)
        self.file_path_browse_button.setMinimumSize(QSize(0, 0))
        self.file_path_browse_button.setMaximumSize(QSize(32, 16777215))
        self.file_path_browse_button.setAutoFillBackground(False)
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.file_path_browse_button.setIcon(icon)
        self.file_path_browse_button.setIconSize(QSize(16, 16))

        self.gridLayout_4.addWidget(self.file_path_browse_button, 2, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)

        self.image_paths_header = QLabel(self.image_import_frame)
        self.image_paths_header.setObjectName(u"image_paths_header")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.image_paths_header.sizePolicy().hasHeightForWidth())
        self.image_paths_header.setSizePolicy(sizePolicy6)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.image_paths_header.setFont(font)
        self.image_paths_header.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_4.addWidget(self.image_paths_header, 1, 0, 1, 3)


        self.gridLayout_2.addWidget(self.image_import_frame, 0, 3, 1, 1)

        self.controls_frame = QFrame(featureFinder)
        self.controls_frame.setObjectName(u"controls_frame")
        sizePolicy2.setHeightForWidth(self.controls_frame.sizePolicy().hasHeightForWidth())
        self.controls_frame.setSizePolicy(sizePolicy2)
        self.controls_frame.setMinimumSize(QSize(0, 0))
        self.controls_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.controls_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_3 = QGridLayout(self.controls_frame)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.feature_max_size_label = QLabel(self.controls_frame)
        self.feature_max_size_label.setObjectName(u"feature_max_size_label")
        sizePolicy3.setHeightForWidth(self.feature_max_size_label.sizePolicy().hasHeightForWidth())
        self.feature_max_size_label.setSizePolicy(sizePolicy3)
        self.feature_max_size_label.setMinimumSize(QSize(100, 0))
        self.feature_max_size_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.feature_max_size_label, 5, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.threshold_spin = QSpinBox(self.controls_frame)
        self.threshold_spin.setObjectName(u"threshold_spin")
        sizePolicy6.setHeightForWidth(self.threshold_spin.sizePolicy().hasHeightForWidth())
        self.threshold_spin.setSizePolicy(sizePolicy6)
        self.threshold_spin.setMinimumSize(QSize(100, 0))
        self.threshold_spin.setMaximum(255)
        self.threshold_spin.setSingleStep(1)

        self.gridLayout_3.addWidget(self.threshold_spin, 2, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.rect_detection_label = QLabel(self.controls_frame)
        self.rect_detection_label.setObjectName(u"rect_detection_label")
        sizePolicy3.setHeightForWidth(self.rect_detection_label.sizePolicy().hasHeightForWidth())
        self.rect_detection_label.setSizePolicy(sizePolicy3)
        self.rect_detection_label.setMinimumSize(QSize(100, 0))
        self.rect_detection_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.rect_detection_label, 1, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_min_size_spin = QSpinBox(self.controls_frame)
        self.blob_min_size_spin.setObjectName(u"blob_min_size_spin")
        sizePolicy3.setHeightForWidth(self.blob_min_size_spin.sizePolicy().hasHeightForWidth())
        self.blob_min_size_spin.setSizePolicy(sizePolicy3)
        self.blob_min_size_spin.setMinimumSize(QSize(100, 0))
        self.blob_min_size_spin.setMaximum(100000)
        self.blob_min_size_spin.setSingleStep(100)

        self.gridLayout_3.addWidget(self.blob_min_size_spin, 9, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_max_size_slider = QSlider(self.controls_frame)
        self.blob_max_size_slider.setObjectName(u"blob_max_size_slider")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.blob_max_size_slider.sizePolicy().hasHeightForWidth())
        self.blob_max_size_slider.setSizePolicy(sizePolicy7)
        self.blob_max_size_slider.setMinimumSize(QSize(250, 0))
        self.blob_max_size_slider.setMaximum(100000)
        self.blob_max_size_slider.setSingleStep(100)
        self.blob_max_size_slider.setValue(100000)
        self.blob_max_size_slider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_3.addWidget(self.blob_max_size_slider, 10, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.feature_max_size_slider = QSlider(self.controls_frame)
        self.feature_max_size_slider.setObjectName(u"feature_max_size_slider")
        sizePolicy7.setHeightForWidth(self.feature_max_size_slider.sizePolicy().hasHeightForWidth())
        self.feature_max_size_slider.setSizePolicy(sizePolicy7)
        self.feature_max_size_slider.setMinimumSize(QSize(250, 0))
        self.feature_max_size_slider.setMaximum(100000)
        self.feature_max_size_slider.setSingleStep(100)
        self.feature_max_size_slider.setValue(100000)
        self.feature_max_size_slider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_3.addWidget(self.feature_max_size_slider, 5, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.verticalSpacer = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.gridLayout_3.addItem(self.verticalSpacer, 6, 1, 1, 1)

        self.threshold_label = QLabel(self.controls_frame)
        self.threshold_label.setObjectName(u"threshold_label")
        sizePolicy3.setHeightForWidth(self.threshold_label.sizePolicy().hasHeightForWidth())
        self.threshold_label.setSizePolicy(sizePolicy3)
        self.threshold_label.setMinimumSize(QSize(100, 0))
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.threshold_label, 2, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.circularity_spin = QDoubleSpinBox(self.controls_frame)
        self.circularity_spin.setObjectName(u"circularity_spin")
        sizePolicy3.setHeightForWidth(self.circularity_spin.sizePolicy().hasHeightForWidth())
        self.circularity_spin.setSizePolicy(sizePolicy3)
        self.circularity_spin.setMinimumSize(QSize(100, 0))
        self.circularity_spin.setMinimum(0.100000000000000)
        self.circularity_spin.setMaximum(1.000000000000000)
        self.circularity_spin.setSingleStep(0.100000000000000)

        self.gridLayout_3.addWidget(self.circularity_spin, 8, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_max_size_label = QLabel(self.controls_frame)
        self.blob_max_size_label.setObjectName(u"blob_max_size_label")
        sizePolicy3.setHeightForWidth(self.blob_max_size_label.sizePolicy().hasHeightForWidth())
        self.blob_max_size_label.setSizePolicy(sizePolicy3)
        self.blob_max_size_label.setMinimumSize(QSize(100, 0))
        self.blob_max_size_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.blob_max_size_label, 10, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.gauss_blur_spin = QSpinBox(self.controls_frame)
        self.gauss_blur_spin.setObjectName(u"gauss_blur_spin")
        sizePolicy3.setHeightForWidth(self.gauss_blur_spin.sizePolicy().hasHeightForWidth())
        self.gauss_blur_spin.setSizePolicy(sizePolicy3)
        self.gauss_blur_spin.setMinimumSize(QSize(100, 0))
        self.gauss_blur_spin.setMinimum(1)
        self.gauss_blur_spin.setMaximum(200)
        self.gauss_blur_spin.setSingleStep(1)

        self.gridLayout_3.addWidget(self.gauss_blur_spin, 3, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_min_size_label = QLabel(self.controls_frame)
        self.blob_min_size_label.setObjectName(u"blob_min_size_label")
        sizePolicy3.setHeightForWidth(self.blob_min_size_label.sizePolicy().hasHeightForWidth())
        self.blob_min_size_label.setSizePolicy(sizePolicy3)
        self.blob_min_size_label.setMinimumSize(QSize(100, 0))
        self.blob_min_size_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.blob_min_size_label, 9, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.crosshair_detection_check = QCheckBox(self.controls_frame)
        self.crosshair_detection_check.setObjectName(u"crosshair_detection_check")
        sizePolicy7.setHeightForWidth(self.crosshair_detection_check.sizePolicy().hasHeightForWidth())
        self.crosshair_detection_check.setSizePolicy(sizePolicy7)
        self.crosshair_detection_check.setChecked(True)

        self.gridLayout_3.addWidget(self.crosshair_detection_check, 1, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_subheader_label = QLabel(self.controls_frame)
        self.blob_subheader_label.setObjectName(u"blob_subheader_label")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.blob_subheader_label.sizePolicy().hasHeightForWidth())
        self.blob_subheader_label.setSizePolicy(sizePolicy8)
        font1 = QFont()
        font1.setPointSize(9)
        font1.setBold(False)
        self.blob_subheader_label.setFont(font1)
        self.blob_subheader_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.blob_subheader_label, 7, 0, 1, 3)

        self.gauss_blur_slider = QSlider(self.controls_frame)
        self.gauss_blur_slider.setObjectName(u"gauss_blur_slider")
        sizePolicy7.setHeightForWidth(self.gauss_blur_slider.sizePolicy().hasHeightForWidth())
        self.gauss_blur_slider.setSizePolicy(sizePolicy7)
        self.gauss_blur_slider.setMinimumSize(QSize(255, 0))
        self.gauss_blur_slider.setMinimum(1)
        self.gauss_blur_slider.setMaximum(200)
        self.gauss_blur_slider.setSingleStep(1)
        self.gauss_blur_slider.setValue(1)
        self.gauss_blur_slider.setOrientation(Qt.Orientation.Horizontal)
        self.gauss_blur_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.gauss_blur_slider.setTickInterval(0)

        self.gridLayout_3.addWidget(self.gauss_blur_slider, 3, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.gauss_blur_label = QLabel(self.controls_frame)
        self.gauss_blur_label.setObjectName(u"gauss_blur_label")
        sizePolicy3.setHeightForWidth(self.gauss_blur_label.sizePolicy().hasHeightForWidth())
        self.gauss_blur_label.setSizePolicy(sizePolicy3)
        self.gauss_blur_label.setMinimumSize(QSize(100, 0))
        self.gauss_blur_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.gauss_blur_label, 3, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.feature_min_size_label = QLabel(self.controls_frame)
        self.feature_min_size_label.setObjectName(u"feature_min_size_label")
        sizePolicy3.setHeightForWidth(self.feature_min_size_label.sizePolicy().hasHeightForWidth())
        self.feature_min_size_label.setSizePolicy(sizePolicy3)
        self.feature_min_size_label.setMinimumSize(QSize(100, 0))
        self.feature_min_size_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.feature_min_size_label, 4, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_max_size_spin = QSpinBox(self.controls_frame)
        self.blob_max_size_spin.setObjectName(u"blob_max_size_spin")
        sizePolicy3.setHeightForWidth(self.blob_max_size_spin.sizePolicy().hasHeightForWidth())
        self.blob_max_size_spin.setSizePolicy(sizePolicy3)
        self.blob_max_size_spin.setMinimumSize(QSize(100, 0))
        self.blob_max_size_spin.setMaximum(100000)
        self.blob_max_size_spin.setSingleStep(100)
        self.blob_max_size_spin.setValue(100000)

        self.gridLayout_3.addWidget(self.blob_max_size_spin, 10, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.feature_max_size_spin = QSpinBox(self.controls_frame)
        self.feature_max_size_spin.setObjectName(u"feature_max_size_spin")
        sizePolicy3.setHeightForWidth(self.feature_max_size_spin.sizePolicy().hasHeightForWidth())
        self.feature_max_size_spin.setSizePolicy(sizePolicy3)
        self.feature_max_size_spin.setMinimumSize(QSize(100, 0))
        self.feature_max_size_spin.setMaximum(100000)
        self.feature_max_size_spin.setSingleStep(100)
        self.feature_max_size_spin.setValue(100000)

        self.gridLayout_3.addWidget(self.feature_max_size_spin, 5, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.blob_min_size_slider = QSlider(self.controls_frame)
        self.blob_min_size_slider.setObjectName(u"blob_min_size_slider")
        sizePolicy7.setHeightForWidth(self.blob_min_size_slider.sizePolicy().hasHeightForWidth())
        self.blob_min_size_slider.setSizePolicy(sizePolicy7)
        self.blob_min_size_slider.setMinimumSize(QSize(250, 0))
        self.blob_min_size_slider.setMaximum(100000)
        self.blob_min_size_slider.setSingleStep(100)
        self.blob_min_size_slider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_3.addWidget(self.blob_min_size_slider, 9, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.threshold_slider = QSlider(self.controls_frame)
        self.threshold_slider.setObjectName(u"threshold_slider")
        sizePolicy7.setHeightForWidth(self.threshold_slider.sizePolicy().hasHeightForWidth())
        self.threshold_slider.setSizePolicy(sizePolicy7)
        self.threshold_slider.setMinimumSize(QSize(255, 0))
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setOrientation(Qt.Orientation.Horizontal)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.threshold_slider.setTickInterval(0)

        self.gridLayout_3.addWidget(self.threshold_slider, 2, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.circularity_label = QLabel(self.controls_frame)
        self.circularity_label.setObjectName(u"circularity_label")
        sizePolicy3.setHeightForWidth(self.circularity_label.sizePolicy().hasHeightForWidth())
        self.circularity_label.setSizePolicy(sizePolicy3)
        self.circularity_label.setMinimumSize(QSize(100, 0))
        self.circularity_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.circularity_label, 8, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.feature_min_size_slider = QSlider(self.controls_frame)
        self.feature_min_size_slider.setObjectName(u"feature_min_size_slider")
        sizePolicy7.setHeightForWidth(self.feature_min_size_slider.sizePolicy().hasHeightForWidth())
        self.feature_min_size_slider.setSizePolicy(sizePolicy7)
        self.feature_min_size_slider.setMinimumSize(QSize(255, 0))
        self.feature_min_size_slider.setMaximum(100000)
        self.feature_min_size_slider.setSingleStep(100)
        self.feature_min_size_slider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_3.addWidget(self.feature_min_size_slider, 4, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.circularity_slider = QSlider(self.controls_frame)
        self.circularity_slider.setObjectName(u"circularity_slider")
        sizePolicy7.setHeightForWidth(self.circularity_slider.sizePolicy().hasHeightForWidth())
        self.circularity_slider.setSizePolicy(sizePolicy7)
        self.circularity_slider.setMinimumSize(QSize(250, 0))
        self.circularity_slider.setMinimum(10)
        self.circularity_slider.setMaximum(100)
        self.circularity_slider.setSingleStep(1)
        self.circularity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.circularity_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.circularity_slider.setTickInterval(0)

        self.gridLayout_3.addWidget(self.circularity_slider, 8, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.detection_controls_header = QLabel(self.controls_frame)
        self.detection_controls_header.setObjectName(u"detection_controls_header")
        sizePolicy6.setHeightForWidth(self.detection_controls_header.sizePolicy().hasHeightForWidth())
        self.detection_controls_header.setSizePolicy(sizePolicy6)
        self.detection_controls_header.setFont(font)
        self.detection_controls_header.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.detection_controls_header, 0, 0, 1, 3)

        self.feature_min_size_spin = QSpinBox(self.controls_frame)
        self.feature_min_size_spin.setObjectName(u"feature_min_size_spin")
        sizePolicy3.setHeightForWidth(self.feature_min_size_spin.sizePolicy().hasHeightForWidth())
        self.feature_min_size_spin.setSizePolicy(sizePolicy3)
        self.feature_min_size_spin.setMinimumSize(QSize(100, 0))
        self.feature_min_size_spin.setMaximum(100000)
        self.feature_min_size_spin.setSingleStep(100)

        self.gridLayout_3.addWidget(self.feature_min_size_spin, 4, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)


        self.gridLayout_2.addWidget(self.controls_frame, 1, 3, 1, 1)


        self.retranslateUi(featureFinder)

        QMetaObject.connectSlotsByName(featureFinder)
    # setupUi

    def retranslateUi(self, featureFinder):
        featureFinder.setWindowTitle(QCoreApplication.translate("featureFinder", u"Feature Finder", None))
        self.file_path_label.setText(QCoreApplication.translate("featureFinder", u"Input File Path:", None))
        self.save_image_label.setText(QCoreApplication.translate("featureFinder", u"Export drawing:", None))
#if QT_CONFIG(tooltip)
        self.save_image_button.setToolTip(QCoreApplication.translate("featureFinder", u"Save the displayed image to provided path", None))
#endif // QT_CONFIG(tooltip)
        self.save_image_button.setText(QCoreApplication.translate("featureFinder", u"Save to File...", None))
#if QT_CONFIG(tooltip)
        self.file_path_entry.setToolTip(QCoreApplication.translate("featureFinder", u"Path to input file ", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.file_path_browse_button.setToolTip(QCoreApplication.translate("featureFinder", u"Path browser for input file ", None))
#endif // QT_CONFIG(tooltip)
        self.file_path_browse_button.setText("")
        self.image_paths_header.setText(QCoreApplication.translate("featureFinder", u"Image Import/Export", None))
        self.feature_max_size_label.setText(QCoreApplication.translate("featureFinder", u"Max. Size:", None))
#if QT_CONFIG(tooltip)
        self.threshold_spin.setToolTip(QCoreApplication.translate("featureFinder", u"Threshold value for edge detection", None))
#endif // QT_CONFIG(tooltip)
        self.rect_detection_label.setText(QCoreApplication.translate("featureFinder", u"Rect. Detection:", None))
#if QT_CONFIG(tooltip)
        self.feature_max_size_slider.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.threshold_label.setText(QCoreApplication.translate("featureFinder", u"Pixel Threshold:", None))
#if QT_CONFIG(tooltip)
        self.circularity_spin.setToolTip(QCoreApplication.translate("featureFinder", u"The closer to 1, the more \"perfect\" the circle is", None))
#endif // QT_CONFIG(tooltip)
        self.blob_max_size_label.setText(QCoreApplication.translate("featureFinder", u"Max. Size:", None))
#if QT_CONFIG(tooltip)
        self.gauss_blur_spin.setToolTip(QCoreApplication.translate("featureFinder", u"Gaussian blur kernel size", None))
#endif // QT_CONFIG(tooltip)
        self.blob_min_size_label.setText(QCoreApplication.translate("featureFinder", u"Min. Size:", None))
#if QT_CONFIG(tooltip)
        self.crosshair_detection_check.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.crosshair_detection_check.setText(QCoreApplication.translate("featureFinder", u"On / Off (Crosshair Detection)", None))
        self.blob_subheader_label.setText(QCoreApplication.translate("featureFinder", u"Blob-Specific Controls...", None))
#if QT_CONFIG(tooltip)
        self.gauss_blur_slider.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.gauss_blur_label.setText(QCoreApplication.translate("featureFinder", u"Gaussian Blur:", None))
        self.feature_min_size_label.setText(QCoreApplication.translate("featureFinder", u"Min. Size:", None))
#if QT_CONFIG(tooltip)
        self.threshold_slider.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.circularity_label.setText(QCoreApplication.translate("featureFinder", u"Circularity:", None))
#if QT_CONFIG(tooltip)
        self.feature_min_size_slider.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.circularity_slider.setToolTip(QCoreApplication.translate("featureFinder", u"The closer to 1, the more \"perfect\" the circle is", None))
#endif // QT_CONFIG(tooltip)
        self.detection_controls_header.setText(QCoreApplication.translate("featureFinder", u"Detection Controls", None))
    # retranslateUi

