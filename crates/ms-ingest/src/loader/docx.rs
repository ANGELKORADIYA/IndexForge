use super::ExtractedDoc;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use zip::ZipArchive;
use quick_xml::events::Event;
use quick_xml::Reader;

pub fn load(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut doc_xml = archive.by_name("word/document.xml")?;
    let mut xml_content = String::new();
    doc_xml.read_to_string(&mut xml_content)?;

    let mut reader = Reader::from_str(&xml_content);
    reader.trim_text(true);

    let mut text = String::new();
    let mut in_text_node = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"w:t" => {
                in_text_node = true;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"w:t" => {
                in_text_node = false;
                text.push(' ');
            }
            Ok(Event::Text(e)) if in_text_node => {
                if let Ok(t) = e.unescape() {
                    text.push_str(&t);
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break, // Graceful degradation
            _ => (),
        }
        buf.clear();
    }

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedDoc {
        source: path.display().to_string(),
        text: text.trim().to_string(),
    }])
}
