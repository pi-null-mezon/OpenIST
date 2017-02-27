import QtQuick 2.7
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0

ApplicationWindow {
    id: appWindow
    visible: true
    width: 640
    height: 480
    title: qsTr("QIMarker")
    color: "dark grey"


    //---------------------------------
    signal dirUrlChanged(url _dirurl)
    signal askNextFile()
    signal askMarksSave()

    //---------------------------------
    function updateImage(_fileurl) {
        currentImage.source = _fileurl
        fileUrl.text = _fileurl
    }
    //---------------------------------

    footer: ToolBar {

        RowLayout {
            spacing: 10
            anchors.fill: parent

            ToolButton {
                id: saveimgButton
                Layout.alignment: Qt.AlignLeft
                text: "Next"
                font.pixelSize: 14
                onClicked: appWindow.askNextFile()
            }

            Label {
                id: fileUrl
                font.pixelSize: 12
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignHCenter
                elide: Text.ElideMiddle
                text: currentImage.source
            }            

            ToolButton {
                id: nextimgButton
                Layout.alignment: Qt.AlignRight
                text: "Save"
                font.pixelSize: 14
                onClicked: askMarksSave()
            }
        }
    }

    header: ToolBar {

        RowLayout {
            spacing: 0
            anchors.fill: parent

            ToolButton {
                contentItem: Image {
                    fillMode: Image.Pad
                    horizontalAlignment: Image.AlignHCenter
                    verticalAlignment: Image.AlignVCenter
                    source: "qrc:/Resources/Images/drawer.png"
                }
                onClicked: fileDialog.open()

                ToolTip.timeout: 10000
                ToolTip.visible: pressed
                ToolTip.text: "Select directory with images"
            }

           Label {
                id: titleLabel
                text: qsTr("QIMarker")
                font.pixelSize: 12
                elide: Label.ElideRight
                horizontalAlignment: Qt.AlignHCenter
                verticalAlignment: Qt.AlignVCenter
                Layout.fillWidth: true
            }

            ToolButton {
                contentItem: Image {
                    fillMode: Image.Pad
                    horizontalAlignment: Image.AlignHCenter
                    verticalAlignment: Image.AlignVCenter
                    source: "qrc:/Resources/Images/menu.png"
                }
                onClicked: optionsMenu.open()

                Menu {
                    id: optionsMenu
                    MenuItem {                        
                        text: "About"
                        onTriggered: aboutDialog.open()
                    }
                    MenuItem {
                        text: "Close"
                        onTriggered: appWindow.close()
                    }
                }
            }
        }
    }

    RowLayout {
        spacing: 0
        anchors.fill: parent

        Rectangle {
            Layout.margins: 5
            Layout.fillWidth: true
            Layout.fillHeight: true

            Image {
                id: currentImage
                anchors.fill: parent

                source: "qrc:/Resources/Images/Startup.png"

                fillMode: Image.PreserveAspectFit
            }
        }


        Rectangle {
            color: "light grey"
            Layout.margins: 5
            Layout.fillWidth: true
            Layout.fillHeight: true


            ListView {
                id: listView
                anchors.fill: parent
                model: classList
                spacing: 10

                delegate: CheckBox {
                    id: checkBox
                    text: model.modelData.name
                    font.pixelSize: 14
                    checked: model.modelData.checked
                    onCheckStateChanged: model.modelData.checked = checkBox.checked
                }

                ScrollIndicator.vertical: ScrollIndicator { }
            }

        }

    }



    FileDialog {
        id: fileDialog
        selectFolder: true

        onAccepted: {
            appWindow.dirUrlChanged(fileDialog.folder)
        }
    }

    Popup {
        id: aboutDialog
        modal: true
        focus: true        
        width: appWindow.width / 2
        x: (appWindow.width - width) / 2
        y: parent.height / 6
        contentHeight: aboutColumn.height

        Column {
            id: aboutColumn
            spacing: 20

            Label {
                text: "About"
                font.bold: true
            }

            Label {
                width: aboutDialog.availableWidth
                text: "QIMarker"
                wrapMode: Label.Wrap
                font.pixelSize: 12
            }

            Label {
                width: aboutDialog.availableWidth
                text: "Application was designed for the automatization of the images markup process."
                    + " Create Labels.ini file with desired labels and drop it to the app.exe location"
                    + " run app, select directory with raw images and start to label. All labeled images"
                    + " will be saved in the subdirectory with name Marked. Then take a coffee brake..."
                wrapMode: Label.Wrap
                font.pixelSize: 12
            }

            Label {
                width: aboutDialog.availableWidth
                text: "Designed by Alex.A.Taranov, 2016, Qt"
                wrapMode: Label.Wrap
                font.pixelSize: 12
            }
        }
    }
}
