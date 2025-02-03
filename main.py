import sys
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import json
import shutil
import pyperclip
import numpy as np
import trimesh
import vtk
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QLineEdit, QComboBox, QScrollArea, QFrame, QDialog,
                           QDialogButtonBox, QMessageBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

@dataclass
class Model3D:
    """Class representing a 3D model with its metadata"""
    id: str
    name: str
    file_path: str
    upload_date: datetime
    tags: List[str]
    preview_path: str


class ModelViewer(QWidget):
    """A widget that displays model previews using VTK"""

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(QSize(256, 256))
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)

        # Generate and set the preview image
        self.generate_preview()

    def generate_preview(self):
        """Generate a static preview image of the 3D model"""
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(256, 256)

        renderer = vtk.vtkRenderer()
        render_window.AddRenderer(renderer)

        extension = os.path.splitext(self.model_path)[1].lower()
        if extension == '.obj':
            reader = vtk.vtkOBJReader()
        elif extension == '.stl':
            reader = vtk.vtkSTLReader()
        else:
            return

        reader.SetFileName(self.model_path)
        reader.Update()

        # Set up the visualization pipeline
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(0.3, 0.5, 0.7)
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.2)

        renderer.AddActor(actor)
        renderer.ResetCamera()

        camera = renderer.GetActiveCamera()
        camera.Elevation(30)
        camera.Azimuth(30)
        renderer.ResetCamera()

        light = vtk.vtkLight()
        light.SetPosition(1, 1, 1)
        light.SetIntensity(0.8)
        renderer.AddLight(light)

        render_window.Render()

        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.Update()

        vtk_image = window_to_image.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        array_buffer = memoryview(vtk_array)

        image = QImage(array_buffer, width, height, QImage.Format.Format_RGB888)

        image = image.mirrored(horizontal=False, vertical=True)

        pixmap = QPixmap.fromImage(image)
        self.preview_label.setPixmap(pixmap)

        # Clean up VTK objects
        render_window.Finalize()
        del render_window
        del renderer
        del actor
        del mapper
        del reader


class TagDialog(QDialog):
    """A dialog window for editing model tags"""

    def __init__(self, current_tags, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Tags")
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }

            QLabel {
                color: #ffffff;
                font-size: 14px;
            }

            QLineEdit {
                background-color: #424242;
                color: white;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
                margin: 4px 0px;
            }

            QLineEdit:focus {
                border: 1px solid #0d47a1;
            }

            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }

            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        self.setup_ui(current_tags)

    def setup_ui(self, current_tags):
        layout = QVBoxLayout(self)

        self.tag_input = QLineEdit(self)
        self.tag_input.setText(", ".join(current_tags))
        self.tag_input.setPlaceholderText("Enter tags separated by commas")

        # Add labels and the input field to the layout
        layout.addWidget(QLabel("Tags:"))
        layout.addWidget(self.tag_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_tags(self) -> List[str]:
        tags_text = self.tag_input.text()
        return [tag.strip() for tag in tags_text.split(",") if tag.strip()]


class ModelDatabase:
    """Manages storage and retrieval of 3D models and their metadata"""

    def __init__(self, storage_path: str):
        # Set up storage directories
        self.storage_path = storage_path
        self.models_path = os.path.join(storage_path, "models")
        self.previews_path = os.path.join(storage_path, "previews")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        self.models: List[Model3D] = []

        # Create necessary directories if they don't exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.previews_path, exist_ok=True)

        self._load_metadata()

    def _load_metadata(self):
        """Load model metadata from JSON file"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.models = [
                    Model3D(
                        id=model['id'],
                        name=model['name'],
                        file_path=model['file_path'],
                        upload_date=datetime.fromisoformat(model['upload_date']),
                        tags=model['tags'],
                        preview_path=model['preview_path']
                    )
                    for model in data
                ]

    def _save_metadata(self):
        """Save model metadata to JSON file"""
        data = [
            {
                'id': model.id,
                'name': model.name,
                'file_path': model.file_path,
                'upload_date': model.upload_date.isoformat(),
                'tags': model.tags,
                'preview_path': model.preview_path
            }
            for model in self.models
        ]
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_model(self, file_path: str, tags: List[str]) -> Model3D:
        """Add a new model to the database"""
        model_id = str(len(self.models) + 1)
        name = os.path.basename(file_path)
        new_file_path = os.path.join(self.models_path, f"{model_id}_{name}")
        preview_path = os.path.join(self.previews_path, f"{model_id}.png")

        shutil.copy2(file_path, new_file_path)

        model = Model3D(
            id=model_id,
            name=name,
            file_path=new_file_path,
            upload_date=datetime.now(),
            tags=tags,
            preview_path=preview_path
        )

        self.models.append(model)
        self._save_metadata()
        return model

    def delete_model(self, model_id: str):
        """Delete a model from the database"""
        model = next((m for m in self.models if m.id == model_id), None)
        if model:
            if os.path.exists(model.file_path):
                os.remove(model.file_path)
            if os.path.exists(model.preview_path):
                os.remove(model.preview_path)

            self.models = [m for m in self.models if m.id != model_id]
            self._save_metadata()

    def get_model_content(self, model_id: str) -> Optional[str]:
        """Get the content of a model file"""
        model = next((m for m in self.models if m.id == model_id), None)
        if model and os.path.exists(model.file_path):
            with open(model.file_path, 'r') as f:
                return f.read()
        return None

    def update_tags(self, model_id: str, tags: List[str]):
        """Update the tags for a model"""
        model = next((m for m in self.models if m.id == model_id), None)
        if model:
            model.tags = tags
            self._save_metadata()


class DetailViewer(QDialog):
    """A dialog for interactive 3D viewing"""

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Model Viewer")
        self.setModal(True)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self.load_model(model_path)

        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.iren.Initialize()
        self.renderer.ResetCamera()

    def load_model(self, model_path):
        """Load and set up the 3D model for interactive viewing"""
        extension = os.path.splitext(model_path)[1].lower()
        if extension == '.obj':
            reader = vtk.vtkOBJReader()
        elif extension == '.stl':
            reader = vtk.vtkSTLReader()
        else:
            return

        reader.SetFileName(model_path)
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(0.3, 0.5, 0.7)
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.2)

        self.renderer.AddActor(actor)

        light = vtk.vtkLight()
        light.SetPosition(1, 1, 1)
        light.SetIntensity(0.8)
        self.renderer.AddLight(light)

    def closeEvent(self, event):
        self.vtk_widget.Finalize()
        super().closeEvent(event)


class ModelCard(QFrame):
    """Widget representing a single model in the gallery"""

    def __init__(self, model: Model3D, parent=None):
        super().__init__(parent)
        self.model = model
        self.parent_window = parent
        self.setStyleSheet("""
            ModelCard {
                background-color: #333333;
                border-radius: 8px;
                padding: 8px;
            }

            QLabel {
                color: #ffffff;
                padding: 2px;
            }

            QPushButton {
                min-width: 70px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        """Set up the card's user interface"""
        layout = QVBoxLayout()
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

        self.viewer = ModelViewer(self.model.file_path)
        self.viewer.setFixedSize(QSize(256, 256))
        self.viewer.setCursor(Qt.CursorShape.PointingHandCursor)
        self.viewer.mousePressEvent = self.show_detail_viewer
        layout.addWidget(self.viewer)

        # Set up the information section
        info_layout = QVBoxLayout()
        self.name_label = QLabel(f"Name: {self.model.name}")
        self.date_label = QLabel(f"Upload Date: {self.model.upload_date.strftime('%Y-%m-%d')}")
        self.tags_label = QLabel(f"Tags: {', '.join(self.model.tags)}")
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.date_label)
        info_layout.addWidget(self.tags_label)

        button_layout = QHBoxLayout()
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_model)
        edit_tags_button = QPushButton("Edit Tags")
        edit_tags_button.clicked.connect(self.edit_tags)

        button_layout.addWidget(copy_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(edit_tags_button)

        # Add all components to the main layout
        layout.addLayout(info_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def show_detail_viewer(self, event):
        viewer = DetailViewer(self.model.file_path, self)
        viewer.exec()

    def copy_to_clipboard(self):
        content = self.parent_window.db.get_model_content(self.model.id)
        if content:
            pyperclip.copy(content)
            QMessageBox.information(self, "Success", "Model content copied to clipboard!")
        else:
            QMessageBox.warning(self, "Error", "Could not read model content!")

    def delete_model(self):
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {self.model.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Delete the model from the database
            self.parent_window.db.delete_model(self.model.id)
            # Remove this card from the interface
            self.deleteLater()

    def edit_tags(self):
        """Open dialog to edit model tags"""
        dialog = TagDialog(self.model.tags, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_tags = dialog.get_tags()
            self.parent_window.db.update_tags(self.model.id, new_tags)
            self.model.tags = new_tags
            # Update just the tags label
            self.tags_label.setText(f"Tags: {', '.join(new_tags)}")

            # Only resort if we're sorting by tags
            if self.parent_window.sort_combo.currentText() == "Tags":
                self.parent_window.resort_gallery()


class GalleryWindow(QMainWindow):
    """Main window of the 3D model gallery application"""

    def __init__(self):
        super().__init__()
        self.db = ModelDatabase("gallery_storage")

        self.setStyleSheet("""
            /* Main window background */
            QMainWindow {
                background-color: #2b2b2b;
            }

            /* All widgets background */
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }

            /* Buttons */
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #1565c0;
            }

            QPushButton:pressed {
                background-color: #0a3880;
            }

            /* ComboBox (dropdown) styling */
            QComboBox {
                background-color: #424242;
                color: white;
                border: 1px solid #555555;
                padding: 6px;
                border-radius: 4px;
            }

            QComboBox:drop-down {
                border: none;
            }

            QComboBox:down-arrow {
                width: 12px;
                height: 12px;
            }

            /* Search/filter input field */
            QLineEdit {
                background-color: #424242;
                color: white;
                border: 1px solid #555555;
                padding: 6px;
                border-radius: 4px;
            }

            QLineEdit:focus {
                border: 1px solid #0d47a1;
            }

            /* Scroll area */
            QScrollArea {
                border: none;
            }

            /* Scrollbar styling */
            QScrollBar:vertical {
                border: none;
                background-color: #2b2b2b;
                width: 10px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background-color: #424242;
                border-radius: 5px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #555555;
            }

            /* Labels */
            QLabel {
                color: #ffffff;
            }
        """)

        self.setup_ui()

    def setup_ui(self):
        """Set up the main window's user interface"""
        self.setWindowTitle("3D Model Gallery")
        self.setMinimumSize(800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        control_panel = QHBoxLayout()

        add_button = QPushButton("Add Model")
        add_button.clicked.connect(self.add_model)

        sort_label = QLabel("Sort by:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Upload Date", "Name", "Tags"])
        self.sort_combo.currentTextChanged.connect(self.resort_gallery)

        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name, tag, or date...")
        self.filter_input.textChanged.connect(self.filter_gallery)

        control_panel.addWidget(add_button)
        control_panel.addWidget(sort_label)
        control_panel.addWidget(self.sort_combo)
        control_panel.addWidget(filter_label)
        control_panel.addWidget(self.filter_input)

        layout.addLayout(control_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(10)
        self.scroll_area.setWidget(self.gallery_widget)
        layout.addWidget(self.scroll_area)

        self.update_gallery()

    def add_model(self):
        """Handle adding a new model to the gallery"""
        # Open file dialog for selecting a 3D model file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D Model",
            "",
            "3D Models (*.obj *.stl)"
        )

        if file_path:
            model = self.db.add_model(file_path, ["default"])
            self.add_model_card(model)

    def add_model_card(self, model):
        """Add a single model card to the gallery"""
        last_row = None
        if self.gallery_layout.count() > 0:
            last_row = self.gallery_layout.itemAt(self.gallery_layout.count() - 1).widget()
            if last_row.layout().count() >= 3:  # If the last row is full
                last_row = None

        if last_row is None:
            last_row = QWidget()
            row_layout = QHBoxLayout(last_row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            self.gallery_layout.addWidget(last_row)

        card = ModelCard(model, self)
        last_row.layout().addWidget(card)

    def resort_gallery(self):
        """Resorts all models in the gallery based on the current sort selection"""
        sort_key = self.sort_combo.currentText()

        models = self.db.models.copy()

        if sort_key == "Upload Date":
            models.sort(key=lambda x: x.upload_date, reverse=True)
        elif sort_key == "Name":
            models.sort(key=lambda x: x.name.lower())
        elif sort_key == "Tags":
            models.sort(key=lambda x: ",".join(sorted(x.tags)))

        self.rearrange_cards(models)

    def filter_gallery(self):
        """Filters the gallery based on the current filter text"""
        filter_text = self.filter_input.text().lower()

        models = self.db.models.copy()

        if filter_text:
            models = [
                m for m in models
                if filter_text in m.name.lower() or
                   filter_text in ",".join(m.tags).lower() or
                   filter_text in m.upload_date.strftime('%Y-%m-%d').lower()
            ]

        self.rearrange_cards(models)

    def rearrange_cards(self, models: List[Model3D]):
        """Rearranges the gallery cards based on the provided model list"""
        existing_cards = {}
        while self.gallery_layout.count() > 0:
            row = self.gallery_layout.takeAt(0).widget()
            if row:
                row_layout = row.layout()
                while row_layout and row_layout.count() > 0:
                    card = row_layout.takeAt(0).widget()
                    if isinstance(card, ModelCard):
                        model_key = (card.model.id, card.model.file_path)
                        existing_cards[model_key] = card
                row.deleteLater()

        current_row = None
        card_count = 0

        for model in models:
            if card_count % 3 == 0:
                current_row = QWidget()
                row_layout = QHBoxLayout(current_row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                self.gallery_layout.addWidget(current_row)

            model_key = (model.id, model.file_path)
            if model_key in existing_cards:
                card = existing_cards.pop(model_key)
            else:
                card = ModelCard(model, self)

            current_row.layout().addWidget(card)
            card_count += 1

        for card in existing_cards.values():
            card.deleteLater()

    def update_gallery(self):
        """Performs a complete gallery update"""
        self.rearrange_cards(self.db.models)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        window = GalleryWindow()
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)