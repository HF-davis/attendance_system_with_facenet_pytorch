import psycopg2
import datetime
def InsertDB(data):

    try:
        connection=psycopg2.connect(
            host="localhost",
            database="Attendance_LAB_SC",
            user="postgres",
            password="123456789",
            port='5432'
        )

        cur=connection.cursor()
        command="INSERT INTO users1(NOMBRE,FECHA,HORA_ENTRADA,HORA_SALIDA,FOTO_ENTRADA,FOTO_SALIDA,STATE) VALUES (%s,%s,%s,%s,%s,%s,%s) "
        cur.execute(command,data)
        connection.commit()
        print("Datos guardados correctamente ")
        
        connection.close()
        
    except (Exception,psycopg2.DatabaseError) as error:
        print(error)

def GetDB(name,fecha):
    try:
        connection=psycopg2.connect(
            host="localhost",
            database="Attendance_LAB_SC",
            user="postgres",
            password="123456789",
            port='5432'
        )

        cur=connection.cursor()
        command="""SELECT * FROM users1 WHERE nombre=%s AND fecha=%s"""
        dat=(name,fecha)
        cur.execute(command,dat)
        result=cur.fetchall()
        print("Los datos obtenidos son: ", result)
            
        connection.close()
        return len(result),result
        
    except (Exception,psycopg2.DatabaseError) as error:
        print(error)



def UpdateDB(name,leave_data,fecha,leave_photo,st):
    try:
        connection=psycopg2.connect(
            host="localhost",
            database="Attendance_LAB_SC",
            user="postgres",
            password="123456789",
            port='5432'
        )

        cur=connection.cursor()
    
        update="""UPDATE users1 SET hora_salida=%s,foto_salida=%s,state=%s WHERE nombre=%s AND fecha=%s"""
        dat=(leave_data,leave_photo,st,name,fecha)
        
        cur.execute(update,dat)
        connection.commit()
        count=cur.rowcount
        print("Record updatede successfully: ", count)
        
        connection.close()
        
    except (Exception,psycopg2.DatabaseError) as error:
        print(error)
